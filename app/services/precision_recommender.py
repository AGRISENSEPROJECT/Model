"""Precision crop ranking — sensor soil + OpenWeatherMap weather."""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

import joblib
import pandas as pd

from app.config import ARTIFACTS_DIR, CANONICAL_CROPS
from app.features.schema import LEGACY_NUMERIC_FEATURE_KEYS, NUMERIC_FEATURE_KEYS
from app.features.vector import enrich_for_crop, merge_domains, provenance_report, split_domains
from app.services.dataset_stats import get_dataset_stats
from app.utils.crops import display_crop_name

logger = logging.getLogger("agrisense.ml")

PRECISION_MODEL_PATH = ARTIFACTS_DIR / "precision_crop_yield.pkl"
PRECISION_META_PATH = ARTIFACTS_DIR / "precision_feature_meta.json"

_lock = threading.Lock()
_bundle: dict[str, Any] | None = None


def artifacts_ready() -> bool:
    return PRECISION_MODEL_PATH.exists() and PRECISION_META_PATH.exists()


def _load() -> dict[str, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is None:
            if not artifacts_ready():
                raise FileNotFoundError(
                    "Precision model missing. Run scripts/train_precision_crop_model.py"
                )
            meta = json.loads(PRECISION_META_PATH.read_text(encoding="utf-8"))
            _bundle = {
                "model": joblib.load(PRECISION_MODEL_PATH),
                "meta": meta,
            }
    return _bundle


def reload_precision_model() -> None:
    global _bundle
    with _lock:
        _bundle = None
    _load()


def _model_numeric_keys(bundle: dict[str, Any]) -> tuple[str, ...]:
    meta_keys = bundle["meta"].get("numeric_features")
    if meta_keys:
        return tuple(meta_keys)
    return NUMERIC_FEATURE_KEYS


def _impute_legacy_ph(features: dict[str, Any], provided_keys: set[str]) -> dict[str, Any]:
    """
    Legacy yield model was trained with soil_ph. Current probe does not measure pH.
    Use training CSV median only for model matrix — flagged in provenance.
    """
    if features.get("soil_ph") is not None:
        return features
    stats = get_dataset_stats()
    # Global N median as rough anchor — prefer explicit pH from CSV global if added
    ph_median = 6.5
    try:
        import pandas as pd

        from app.config import ENV_DATASET_PATH

        if ENV_DATASET_PATH.exists():
            df = pd.read_csv(ENV_DATASET_PATH)
            ph_median = float(df["Soil_pH"].median())
    except Exception:  # noqa: BLE001
        pass
    features = dict(features)
    features["soil_ph"] = ph_median
    provided_keys = set(provided_keys)
    logger.warning(
        "soil_ph not measured — using training median %.2f for legacy ML matrix only",
        ph_median,
    )
    return features


def _rows_for_candidates(
    base_features: dict[str, Any],
    crop_ids: tuple[str, ...],
    numeric_keys: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for crop_id in crop_ids:
        enriched = enrich_for_crop(base_features, crop_id)
        row = {k: enriched.get(k) for k in numeric_keys}
        row["soil_texture"] = enriched.get("soil_texture") or "unknown"
        row["crop_id"] = enriched["crop_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def recommend_crops_precision(
    *,
    soil: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    flat: dict[str, Any] | None = None,
    provided_keys: set[str] | None = None,
    crop_ids: tuple[str, ...] | None = None,
    top_k: int = 5,
    production_mode: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    bundle = _load()
    provided = set(provided_keys or [])
    features = merge_domains(
        soil=soil,
        weather=weather,
        flat=flat,
        production_mode=production_mode,
        allow_weather_defaults=True,
    )

    if features.get("soil_ph") is None:
        features = _impute_legacy_ph(features, provided)

    numeric_keys = _model_numeric_keys(bundle)
    if production_mode and features.get("soil_texture") is None:
        features["soil_texture"] = "unknown"

    candidates = crop_ids or CANONICAL_CROPS
    frame = _rows_for_candidates(features, candidates, numeric_keys)

    logger.info(
        "ML crop ranking input N=%s P=%s K=%s moisture=%s ec=%s soil_temp=%s air_temp=%s",
        features.get("nitrogen"),
        features.get("phosphorus"),
        features.get("potassium"),
        features.get("soil_moisture_vwc"),
        features.get("ec_us_cm"),
        features.get("soil_temperature_c"),
        features.get("temperature_c"),
    )

    preds = bundle["model"].predict(frame)

    ranked = []
    for crop_id, yhat in zip(candidates, preds):
        y = max(0.0, float(yhat))
        ranked.append(
            {
                "crop_id": crop_id,
                "crop": display_crop_name(crop_id),
                "predicted_yield": round(y, 2),
                "source": "precision_ml_yield_ranker",
                "method": "gradient_boosting_regressor",
            }
        )
    ranked.sort(key=lambda x: x["predicted_yield"], reverse=True)
    max_y = max((r["predicted_yield"] for r in ranked), default=1.0) or 1.0
    for r in ranked:
        r["suitability_score"] = round(100.0 * r["predicted_yield"] / max_y, 2)
        r["probability"] = round(r["predicted_yield"] / max_y, 4)

    prov = provenance_report(features, provided)
    if features.get("soil_ph") is not None and "soil_ph" not in provided:
        prov["soil_ph"] = "training_median_imputation"

    return {
        "crop_recommendations": ranked[:top_k],
        "feature_array": split_domains(features),
        "feature_provenance": prov,
        "model_meta": {
            "artifact": str(PRECISION_MODEL_PATH.name),
            "n_features": bundle["meta"].get("n_features"),
            "model_type": bundle["meta"].get("model", "gbr"),
            "training_date": bundle["meta"].get("trained_at"),
            "yield_r2": bundle["meta"].get("yield_r2"),
            "domains": ["soil", "weather"],
        },
    }
