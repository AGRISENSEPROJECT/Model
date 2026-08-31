"""Crop recommendation via RandomForest (live path) + optional rule fallback."""
from __future__ import annotations

import threading
from typing import Any

import joblib
import numpy as np

from app.config import CROP_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH
from app.knowledge.crops import CROP_SUITABILITY
from app.utils.crops import display_crop_name

_lock = threading.Lock()
_bundle: dict[str, Any] | None = None

SUPPORTED_TEXTURES = ("alluvial", "clayey", "loamy", "sandy")


def _load_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is None:
            missing = [
                p.name
                for p in (CROP_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH)
                if not p.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing crop artifacts: {missing}. Run scripts/train_crop_rf.py"
                )
            _bundle = {
                "model": joblib.load(CROP_MODEL_PATH),
                "scaler": joblib.load(SCALER_PATH),
                "label_encoder": joblib.load(LABEL_ENCODER_PATH),
            }
    return _bundle


def reload_bundle() -> None:
    global _bundle
    with _lock:
        _bundle = None
    _load_bundle()


def build_feature_vector(
    soil_texture: str, temperature: float, humidity: float, rainfall: float
) -> np.ndarray:
    """
    Match training pipeline in scripts/train_crop_rf.py:
    features = [encoded_soil_texture, scaled_temp, scaled_humidity, scaled_rainfall]
    NOT one-hot — one-hot would silently break the saved RF.
    """
    bundle = _load_bundle()
    encoder = bundle["label_encoder"]
    scaler = bundle["scaler"]
    if soil_texture not in set(encoder.classes_):
        raise ValueError(
            f"Unsupported soil texture '{soil_texture}'. "
            f"Supported: {sorted(encoder.classes_)}"
        )
    soil_code = encoder.transform([soil_texture])[0]
    scaled = scaler.transform([[temperature, humidity, rainfall]])[0]
    return np.array([[soil_code, scaled[0], scaled[1], scaled[2]]], dtype=float)


def recommend_crops_ml(
    soil_texture: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if soil_texture not in SUPPORTED_TEXTURES:
        raise ValueError(
            f"Unsupported soil texture: '{soil_texture}'. "
            f"Supported: {', '.join(SUPPORTED_TEXTURES)}"
        )
    bundle = _load_bundle()
    model = bundle["model"]
    features = build_feature_vector(soil_texture, temperature, humidity, rainfall)
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "crop_id": crop_id,
            "crop": display_crop_name(crop_id),
            "probability": round(float(prob), 4),
            "suitability_score": round(float(prob) * 100, 2),
            "source": "random_forest",
        }
        for crop_id, prob in ranked
    ]


def recommend_crops_rules(
    soil_texture: str,
    temperature: float,
    humidity: float,
    rainfall: float,
) -> list[dict[str, Any]]:
    """Deterministic checklist used as diagnostics / fallback only."""
    recommendations = []
    for crop_id, reqs in CROP_SUITABILITY.items():
        score = 0.0
        if soil_texture in reqs["soil_texture"]:
            score += 40
        if reqs["temperature"]["min"] <= temperature <= reqs["temperature"]["max"]:
            score += 20
        if reqs["humidity"]["min"] <= humidity <= reqs["humidity"]["max"]:
            score += 20
        if reqs["rainfall"]["min"] <= rainfall <= reqs["rainfall"]["max"]:
            score += 20
        recommendations.append(
            {
                "crop_id": crop_id,
                "crop": display_crop_name(crop_id),
                "suitability_score": score,
                "source": "rules",
            }
        )
    recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
    return recommendations


def recommend_crops(
    soil_texture: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    *,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    soil_ph: float | None = None,
    wind_speed: float | None = None,
    use_ml: bool = True,
    prefer_environmental: bool = True,
) -> list[dict[str, Any]]:
    """
    Production path:
    1) Environmental Factors yield RF when NPK available (real dataset)
    2) Legacy synthetic crop RF
    3) Rule checklist fallback
    """
    if prefer_environmental and use_ml and None not in (nitrogen, phosphorus, potassium):
        try:
            from app.services.environmental import (
                environmental_artifacts_ready,
                recommend_crops_by_yield,
            )

            if environmental_artifacts_ready():
                return recommend_crops_by_yield(
                    soil_texture=soil_texture,
                    temperature=temperature,
                    humidity=humidity,
                    nitrogen=float(nitrogen),
                    phosphorus=float(phosphorus),
                    potassium=float(potassium),
                    soil_ph=soil_ph,
                    wind_speed=wind_speed,
                )
        except Exception:
            pass

    if use_ml:
        try:
            return recommend_crops_ml(soil_texture, temperature, humidity, rainfall)
        except Exception:
            return recommend_crops_rules(soil_texture, temperature, humidity, rainfall)
    return recommend_crops_rules(soil_texture, temperature, humidity, rainfall)
