"""Ingest live soil sensor payloads from ESP32 devices and run server-side AI."""
from __future__ import annotations

import logging
from typing import Any

from app.services.analysis import build_predict_response, run_comprehensive_analysis
from app.services.confidence import compute_recommendation_confidence
from app.services.field_visit_store import save_field_visit
from app.services.sensor_store import save_device_reading
from app.services.sensor_validation import ValidatedSensorReading, validate_sensor_payload

logger = logging.getLogger("agrisense.sensor")


def _reading_to_soil_block(reading: ValidatedSensorReading) -> dict[str, Any]:
    """Map validated probe data to internal soil feature block."""
    block: dict[str, Any] = {
        "soil_temperature_c": reading.temperature_c,
        "soil_moisture_vwc": reading.moisture_pct,
        "ec_us_cm": reading.ec_us_cm,
        "nitrogen": reading.nitrogen,
        "phosphorus": reading.phosphorus,
        "potassium": reading.potassium,
    }
    texture = reading.raw_soil.get("soil_texture")
    if texture:
        block["soil_texture"] = str(texture).strip().lower()
    # Lab pH only — never from RS485 probe
    lab_ph = reading.raw_soil.get("lab_ph") or reading.raw_soil.get("soil_ph_lab")
    if lab_ph is not None:
        try:
            block["soil_ph"] = float(lab_ph)
        except (TypeError, ValueError):
            pass
    return block


def ingest_sensor_reading(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Production path: validate all 6 sensor parameters, run AI, store audit trail.
    """
    logger.info("Raw sensor payload received device=%s", payload.get("device_id"))
    reading = validate_sensor_payload(payload)
    soil_block = _reading_to_soil_block(reading)

    flat = {
        "nitrogen": reading.nitrogen,
        "phosphorus": reading.phosphorus,
        "potassium": reading.potassium,
        "soil_moisture": reading.moisture_pct,
        "ec_us_cm": reading.ec_us_cm,
        "soil_temperature_c": reading.temperature_c,
    }

    analysis = run_comprehensive_analysis(
        nitrogen=reading.nitrogen,
        phosphorus=reading.phosphorus,
        potassium=reading.potassium,
        soil_moisture=reading.moisture_pct,
        temperature=reading.temperature_c,
        coordinates=reading.coordinates,
        soil=soil_block,
        flat=flat,
        production_mode=True,
    )

    predict = build_predict_response(analysis)
    stored_path = save_device_reading(payload, predict)

    confidence = analysis.get("recommendation_confidence") or {}
    visit_path = save_field_visit(
        payload=payload,
        validated_reading={
            "moisture_pct": reading.moisture_pct,
            "temperature_c": reading.temperature_c,
            "ec_us_cm": reading.ec_us_cm,
            "nitrogen": reading.nitrogen,
            "phosphorus": reading.phosphorus,
            "potassium": reading.potassium,
            "coordinates": reading.coordinates,
            "farm_id": reading.farm_id,
        },
        analysis_summary={
            "best_crop": analysis.get("best_crop"),
            "best_crop_id": analysis.get("best_crop_id"),
            "top_crops": (analysis.get("crop_recommendations") or [])[:3],
            "nutrient_analysis": analysis.get("nutrient_analysis"),
        },
        confidence=confidence,
    )

    visit_id = visit_path.name if hasattr(visit_path, "name") else str(visit_path).split("/")[-1]

    crops = predict.get("crop_recommendations") or []
    best = crops[0] if crops else {}

    logger.info(
        "AI result device=%s best_crop=%s score=%s confidence=%s stored=%s visit=%s",
        reading.device_id,
        analysis.get("best_crop"),
        best.get("suitability_score"),
        confidence.get("level"),
        stored_path,
        visit_path,
    )

    return {
        "status": "ok",
        "device_id": reading.device_id,
        "farm_id": reading.farm_id,
        "stored_at": str(stored_path),
        "field_visit_id": visit_id,
        "best_crop": analysis.get("best_crop"),
        "best_crop_id": analysis.get("best_crop_id"),
        "soil_texture": predict.get("soil_texture"),
        "top_crop": best.get("crop"),
        "top_crop_score": best.get("suitability_score"),
        "recommendation_confidence": confidence,
        "pipeline": analysis.get("pipeline"),
        "ai_input": analysis.get("ai_input_log"),
        "analysis": predict,
    }
