"""Honest confidence scoring for production recommendations."""
from __future__ import annotations

from typing import Any


def _level(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    if score >= 25:
        return "low"
    return "very_low"


CNN_CONFIDENCE_FLOOR = 0.55


def compute_recommendation_confidence(
    *,
    feature_provenance: dict[str, str] | None,
    has_weather_api: bool,
    has_soil_texture_cnn: bool,
    soil_texture_source: str | None,
    model_name: str,
    crop_score_spread: float | None = None,
    dataset_calibrated: bool = False,
    soil_texture_confidence: float | None = None,
    soil_texture_label: str | None = None,
) -> dict[str, Any]:
    """
    Score 0–100 based on how much input is live vs defaulted.
    Never claim high confidence when weather or texture are missing.
    """
    score = 100.0
    factors: list[str] = []
    missing: list[str] = []

    prov = feature_provenance or {}
    sensor_keys = (
        "nitrogen",
        "phosphorus",
        "potassium",
        "soil_moisture_vwc",
        "ec_us_cm",
        "soil_temperature_c",
    )
    for key in sensor_keys:
        if prov.get(key) != "live":
            score -= 8
            missing.append(key)

    weather_keys = ("temperature_c", "relative_humidity", "precip_accum_mm", "wind_speed")
    weather_live = sum(1 for k in weather_keys if prov.get(k) == "live")
    if not has_weather_api:
        score -= 20
        missing.append("weather_api")
        factors.append("No live weather — ranking uses soil data primarily")
    elif weather_live < 2:
        score -= 10
        factors.append("Partial weather data")

    if soil_texture_source in ("default_or_request", None) and not has_soil_texture_cnn:
        score -= 15
        missing.append("soil_texture")
        factors.append("Soil texture unknown — add photo scan or lab data")
    elif has_soil_texture_cnn:
        factors.append("Soil texture from image CNN")
        cnn_conf = float(soil_texture_confidence or 0.0)
        if cnn_conf < CNN_CONFIDENCE_FLOOR:
            score -= 18
            factors.append(
                f"Soil CNN confidence {cnn_conf:.0%} is below the {CNN_CONFIDENCE_FLOOR:.0%} production floor"
            )
        if (soil_texture_label or "").lower() == "sandy" and cnn_conf < 0.70:
            score -= 8
            factors.append("Sandy class has historically low recall — treat this texture with caution")

    if not dataset_calibrated:
        score -= 10
        factors.append("Model not calibrated on local Rwanda field yields")

    if crop_score_spread is not None and crop_score_spread < 5:
        score -= 5
        factors.append("Top crops very close in score — recommendation uncertain")

    score = max(0.0, min(100.0, score))
    level = _level(score)

    message = "Recommendation confidence: " + level.replace("_", " ").title()
    if level in ("low", "very_low"):
        message += ". More soil, weather, and farm history data recommended."

    return {
        "score": round(score, 1),
        "level": level,
        "model": model_name,
        "factors": factors,
        "missing_data": missing,
        "message": message,
    }
