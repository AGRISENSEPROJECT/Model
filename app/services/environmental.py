"""Yield-based crop ranking using Environmental Factors models."""
from __future__ import annotations

import threading
from typing import Any

import joblib
import pandas as pd

from app.config import (
    CANONICAL_CROPS,
    QUALITY_MODEL_PATH,
    YIELD_MODEL_PATH,
)
from app.knowledge.environmental_maps import (
    DEFAULT_PH_BY_SOIL,
    DEFAULT_WIND_SPEED,
    ID_TO_DATASET_CROP,
    TEXTURE_TO_DATASET_SOIL,
)
from app.utils.crops import display_crop_name

_lock = threading.Lock()
_bundle: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is None:
            if not YIELD_MODEL_PATH.exists() or not QUALITY_MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Environmental models missing. Run scripts/train_environmental_models.py"
                )
            _bundle = {
                "yield": joblib.load(YIELD_MODEL_PATH),
                "quality": joblib.load(QUALITY_MODEL_PATH),
            }
    return _bundle


def reload_environmental_models() -> None:
    global _bundle
    with _lock:
        _bundle = None
    _load()


def resolve_dataset_soil(soil_texture: str) -> str:
    key = soil_texture.strip().lower()
    if key not in TEXTURE_TO_DATASET_SOIL:
        raise ValueError(
            f"Unsupported soil texture '{soil_texture}'. "
            f"Known: {sorted(TEXTURE_TO_DATASET_SOIL)}"
        )
    return TEXTURE_TO_DATASET_SOIL[key]


def predict_soil_quality(
    soil_texture: str,
    temperature: float,
    humidity: float,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_ph: float | None = None,
    wind_speed: float | None = None,
    crop_id: str = "rice",
) -> dict[str, Any]:
    bundle = _load()
    soil = resolve_dataset_soil(soil_texture)
    ph = float(soil_ph if soil_ph is not None else DEFAULT_PH_BY_SOIL[soil])
    wind = float(wind_speed if wind_speed is not None else DEFAULT_WIND_SPEED)
    crop_label = ID_TO_DATASET_CROP.get(crop_id, "Rice")
    row = pd.DataFrame(
        [
            {
                "Soil_Type": soil,
                "Crop_Type": crop_label,
                "Soil_pH": ph,
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind_Speed": wind,
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
            }
        ]
    )
    score = float(bundle["quality"].predict(row)[0])
    return {
        "soil_quality_score": round(score, 2),
        "dataset_soil_type": soil,
        "soil_ph_used": ph,
        "wind_speed_used": wind,
        "source": "environmental_rf",
    }


def recommend_crops_by_yield(
    soil_texture: str,
    temperature: float,
    humidity: float,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_ph: float | None = None,
    wind_speed: float | None = None,
    crop_ids: tuple[str, ...] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank crops by predicted yield under the current environment."""
    bundle = _load()
    soil = resolve_dataset_soil(soil_texture)
    ph = float(soil_ph if soil_ph is not None else DEFAULT_PH_BY_SOIL[soil])
    wind = float(wind_speed if wind_speed is not None else DEFAULT_WIND_SPEED)
    candidates = crop_ids or CANONICAL_CROPS

    rows = []
    valid_ids = []
    for crop_id in candidates:
        if crop_id not in ID_TO_DATASET_CROP:
            continue
        valid_ids.append(crop_id)
        rows.append(
            {
                "Soil_Type": soil,
                "Crop_Type": ID_TO_DATASET_CROP[crop_id],
                "Soil_pH": ph,
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind_Speed": wind,
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
            }
        )
    if not rows:
        raise ValueError("No overlapping crops between request and environmental dataset")

    frame = pd.DataFrame(rows)
    yields = bundle["yield"].predict(frame)
    max_y = float(max(yields.max(), 1e-6))

    ranked = []
    for crop_id, yhat in zip(valid_ids, yields):
        y = max(0.0, float(yhat))
        ranked.append(
            {
                "crop_id": crop_id,
                "crop": display_crop_name(crop_id),
                "predicted_yield": round(y, 2),
                "suitability_score": round(100.0 * y / max_y, 2),
                "probability": round(y / max_y, 4),
                "source": "environmental_yield_rf",
                "dataset_soil_type": soil,
                "soil_ph_used": ph,
                "wind_speed_used": wind,
            }
        )
    ranked.sort(key=lambda x: x["predicted_yield"], reverse=True)
    return ranked[:top_k]


def environmental_artifacts_ready() -> bool:
    return YIELD_MODEL_PATH.exists() and QUALITY_MODEL_PATH.exists()
