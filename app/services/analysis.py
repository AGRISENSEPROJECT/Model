from __future__ import annotations

from datetime import datetime
from typing import Any

from app.features.vector import merge_domains
from app.services.confidence import compute_recommendation_confidence
from app.services.crop_recommender import recommend_crops
from app.services.disease import detect_disease
from app.services.environmental import (
    environmental_artifacts_ready,
    predict_soil_quality,
)
from app.services.feedback_store import archive_prediction_image
from app.services.fertilizer import recommend_fertilizer
from app.services.irrigation import recommend_irrigation
from app.services.nutrient_analyzer import analyze_nutrients
from app.services.precision_recommender import artifacts_ready as precision_ready
from app.services.precision_recommender import recommend_crops_precision
from app.services.soil_texture import predict_texture
from app.services.weather import fetch_weather_bundle
from app.utils.crops import display_crop_name, normalize_crop_id


def get_satellite_data(coordinates: dict[str, float] | None) -> dict[str, Any] | None:
    if not coordinates:
        return None
    return {
        "status": "optional",
        "coordinates": coordinates,
        "note": "NDVI/satellite still pending; weather uses OpenWeatherMap when lat/lon set",
    }


def _collect_provided_keys(*blocks: dict | None) -> set[str]:
    keys: set[str] = set()
    for block in blocks:
        if not block:
            continue
        keys.update(k for k, v in block.items() if v is not None and v != "")
    return keys


def run_comprehensive_analysis(
    temperature: float | None = None,
    humidity: float | None = None,
    rainfall: float | None = None,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    image_path: str | None = None,
    soil_texture: str | None = None,
    soil_moisture: float | None = None,
    crop_type: str | None = None,
    coordinates: dict[str, float] | None = None,
    soil_ph: float | None = None,
    wind_speed: float | None = None,
    soil: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    flat: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,  # ignored — reserved for future
    economic: dict[str, Any] | None = None,  # ignored
    production_mode: bool = False,
) -> dict[str, Any]:
    """
    Lean path: soil sensors + OpenWeatherMap (when coordinates given).
    production_mode=True: no fake sensor defaults; honest confidence when data missing.
    """
    flat_input: dict[str, Any] = dict(flat or {})
    flat_input.update(
        {
            k: v
            for k, v in {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "soil_moisture": soil_moisture,
                "soil_ph": soil_ph,
                "wind_speed": wind_speed,
                "soil_texture": soil_texture,
            }.items()
            if v is not None
        }
    )

    weather_block = dict(weather or {})
    weather_meta = None
    has_weather_api = False
    if coordinates and coordinates.get("lat") is not None and coordinates.get("lon") is not None:
        try:
            bundle = fetch_weather_bundle(float(coordinates["lat"]), float(coordinates["lon"]))
            for k, v in bundle["weather"].items():
                weather_block.setdefault(k, v)
            weather_meta = bundle["openweathermap"]
            has_weather_api = True
        except Exception as exc:  # noqa: BLE001
            weather_meta = {"error": str(exc), "provider": "openweathermap"}

    soil_block = dict(soil or {})
    feedback_record = None
    has_soil_texture_cnn = False

    if image_path and not soil_texture and "soil_texture" not in soil_block:
        predicted = predict_texture(image_path)
        soil_block["soil_texture"] = predicted["texture"]
        soil_cnn = predicted
        has_soil_texture_cnn = True
        try:
            feedback_record = archive_prediction_image(
                image_path,
                predicted_texture=predicted["texture"],
                confidence=float(predicted.get("confidence") or 0),
                probabilities=predicted.get("probabilities"),
                request_meta={
                    "temperature": flat_input.get("temperature") or weather_block.get("temperature_c"),
                    "humidity": flat_input.get("humidity") or weather_block.get("relative_humidity"),
                    "coordinates": coordinates,
                },
            )
        except Exception:  # noqa: BLE001
            feedback_record = {"error": "archive_failed"}
    elif soil_texture or soil_block.get("soil_texture"):
        texture = str(soil_texture or soil_block.get("soil_texture")).lower()
        soil_block.setdefault("soil_texture", texture)
        soil_cnn = {
            "texture": texture,
            "confidence": 1.0,
            "probabilities": {texture: 1.0},
            "class_index": None,
            "class_index_source": "farmer_or_lab_input",
        }
    else:
        soil_cnn = {
            "texture": None,
            "confidence": 0.0,
            "probabilities": {},
            "class_index": None,
            "class_index_source": "unknown",
        }

    provided = _collect_provided_keys(soil_block, weather_block, flat_input)
    feature_payload = None
    ranking_source = "none"

    merged_preview = merge_domains(
        soil=soil_block,
        weather=weather_block,
        flat=flat_input,
        production_mode=production_mode,
        allow_weather_defaults=not production_mode or not has_weather_api,
    )

    if precision_ready():
        feature_payload = recommend_crops_precision(
            soil=soil_block,
            weather=weather_block,
            flat=flat_input,
            provided_keys=provided,
            production_mode=production_mode,
        )
        crop_recommendations = feature_payload["crop_recommendations"]
        ranking_source = "precision_ml_yield_ranker"
    elif not production_mode:
        texture = soil_cnn.get("texture") or "loamy"
        crop_recommendations = recommend_crops(
            texture,
            float(flat_input.get("temperature") or weather_block.get("temperature_c") or 24),
            float(flat_input.get("humidity") or weather_block.get("relative_humidity") or 70),
            float(flat_input.get("rainfall") or weather_block.get("precip_accum_mm") or 800),
            nitrogen=float(flat_input.get("nitrogen") or 70),
            phosphorus=float(flat_input.get("phosphorus") or 50),
            potassium=float(flat_input.get("potassium") or 40),
            soil_ph=flat_input.get("soil_ph"),
            wind_speed=flat_input.get("wind_speed") or weather_block.get("wind_speed"),
            use_ml=True,
            prefer_environmental=True,
        )
        ranking_source = crop_recommendations[0]["source"] if crop_recommendations else "none"
    else:
        raise RuntimeError(
            "Precision ML model artifacts missing — cannot run production analysis. "
            "Run scripts/train_precision_crop_model.py"
        )

    best = crop_recommendations[0] if crop_recommendations else None
    best_crop_id = best["crop_id"] if best else None
    requested = normalize_crop_id(crop_type) if crop_type else None
    fertilizer_crop = requested or best_crop_id
    irrigation_crop = best_crop_id or requested
    texture = soil_cnn.get("texture") or merged_preview.get("soil_texture")

    nutrient_analysis = None
    if nitrogen is not None and phosphorus is not None and potassium is not None:
        nutrient_analysis = analyze_nutrients(
            nitrogen=float(nitrogen),
            phosphorus=float(phosphorus),
            potassium=float(potassium),
            ec_us_cm=float(soil_block.get("ec_us_cm") or flat_input.get("ec_us_cm") or 0),
            moisture_pct=float(soil_moisture or soil_block.get("soil_moisture_vwc") or 0),
            soil_temperature_c=float(
                soil_block.get("soil_temperature_c") or temperature or 0
            ),
            crop_id=fertilizer_crop,
        )

    fertilizer = (
        recommend_fertilizer(
            texture,
            fertilizer_crop,
            float(nitrogen),
            float(phosphorus),
            float(potassium),
            ec_us_cm=soil_block.get("ec_us_cm"),
            moisture_pct=soil_moisture,
            soil_temperature_c=soil_block.get("soil_temperature_c") or temperature,
        )
        if fertilizer_crop and nitrogen is not None
        else {"message": "Insufficient NPK data for fertilizer analysis"}
    )

    irrigation = (
        recommend_irrigation(
            float(soil_moisture or soil_block.get("soil_moisture_vwc")),
            irrigation_crop,
            float(
                weather_block.get("temperature_c")
                or flat_input.get("temperature")
                or soil_block.get("soil_temperature_c")
                or 24
            ),
            float(weather_block.get("relative_humidity") or flat_input.get("humidity") or 70),
            float(weather_block.get("precip_accum_mm") or flat_input.get("rainfall") or 0),
            soil_texture=texture,
            et0_mm=merged_preview.get("et0_mm"),
        )
        if irrigation_crop and soil_moisture is not None
        else {"message": "Insufficient moisture data for irrigation analysis"}
    )

    disease = detect_disease(image_path or "", irrigation_crop or "unknown")

    env_quality = None
    if environmental_artifacts_ready() and not production_mode:
        try:
            env_quality = predict_soil_quality(
                soil_texture=texture or "loamy",
                temperature=float(weather_block.get("temperature_c") or 24),
                humidity=float(weather_block.get("relative_humidity") or 70),
                nitrogen=float(nitrogen or 0),
                phosphorus=float(phosphorus or 0),
                potassium=float(potassium or 0),
                soil_ph=flat_input.get("soil_ph"),
                wind_speed=weather_block.get("wind_speed"),
                crop_id=irrigation_crop,
            )
        except Exception as exc:  # noqa: BLE001
            env_quality = {"error": str(exc)}

    score_spread = None
    if len(crop_recommendations) >= 2:
        score_spread = (
            crop_recommendations[0]["suitability_score"]
            - crop_recommendations[1]["suitability_score"]
        )

    confidence = compute_recommendation_confidence(
        feature_provenance=(feature_payload or {}).get("feature_provenance"),
        has_weather_api=has_weather_api,
        has_soil_texture_cnn=has_soil_texture_cnn,
        soil_texture_source=soil_cnn.get("class_index_source"),
        model_name=ranking_source,
        crop_score_spread=score_spread,
        dataset_calibrated=False,
    )

    for crop in crop_recommendations:
        crop["confidence_level"] = confidence["level"]
        crop["explanation"] = (
            f"Ranked by ML predicted yield under measured soil conditions "
            f"(confidence: {confidence['level']})."
        )

    ai_input_log = {
        "sensor": split_sensor_log(soil_block, flat_input),
        "weather": weather_block,
        "feature_provenance": (feature_payload or {}).get("feature_provenance"),
        "production_mode": production_mode,
    }

    return {
        "soil_analysis": {
            "texture": texture or "unknown",
            "texture_confidence": soil_cnn.get("confidence"),
            "probabilities": soil_cnn.get("probabilities"),
            "moisture": soil_moisture,
            "ec_us_cm": soil_block.get("ec_us_cm"),
            "soil_temperature_c": soil_block.get("soil_temperature_c"),
            "class_index": soil_cnn.get("class_index"),
            "class_index_source": soil_cnn.get("class_index_source"),
            "environmental_quality": env_quality,
            "ph_measured": flat_input.get("soil_ph") is not None,
            "ph_note": "pH not measured by RS485 probe unless lab_ph provided",
        },
        "nutrient_analysis": nutrient_analysis,
        "crop_recommendations": crop_recommendations,
        "best_crop": display_crop_name(best_crop_id) if best_crop_id else "unknown",
        "best_crop_id": best_crop_id,
        "fertilizer_recommendation": fertilizer,
        "irrigation_recommendation": irrigation,
        "disease_analysis": disease,
        "feature_array": (feature_payload or {}).get("feature_array"),
        "feature_provenance": (feature_payload or {}).get("feature_provenance"),
        "recommendation_confidence": confidence,
        "ai_input_log": ai_input_log,
        "pipeline": [
            "Sensor Reading",
            "Data Validation",
            "Backend Ingest",
            "Feature Processing",
            "ML Crop Ranker",
            "Nutrient & Irrigation Analysis",
            "Crop Recommendation",
        ],
        "weather_forecast": {
            "features": weather_block,
            "openweathermap": weather_meta,
            "note": "Pass coordinates.lat/lon for live weather from OpenWeatherMap",
        },
        "model_stack": {
            "soil_texture": soil_cnn.get("class_index_source"),
            "crop_ranking": ranking_source,
            "precision_ml": precision_ready(),
            "fertilizer": "data_driven_npk_percentiles",
            "irrigation": "dynamic_moisture_et0",
            "disease": "not_trained",
            "domains": ["soil", "weather"],
            "crop_type_provided": bool(requested),
            "hardcoded_rules": False,
        },
        "retrain_capture": feedback_record,
        "satellite_integration": get_satellite_data(coordinates),
        "timestamp": datetime.now().isoformat(),
    }


def split_sensor_log(soil_block: dict[str, Any], flat: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "soil_temperature_c",
        "soil_moisture_vwc",
        "ec_us_cm",
        "nitrogen",
        "phosphorus",
        "potassium",
        "soil_ph",
        "soil_texture",
    )
    out = {}
    for k in keys:
        if soil_block.get(k) is not None:
            out[k] = soil_block[k]
        elif flat.get(k) is not None:
            out[k] = flat[k]
    return out


def build_predict_response(analysis: dict[str, Any]) -> dict[str, Any]:
    crops = analysis["crop_recommendations"]
    irrigation = analysis["irrigation_recommendation"]
    disease = analysis["disease_analysis"]
    fertilizer = analysis["fertilizer_recommendation"]
    confidence = analysis.get("recommendation_confidence") or {}

    return {
        "soil_texture": analysis["soil_analysis"].get("texture") or "unknown",
        "soil_analysis": analysis["soil_analysis"],
        "nutrient_analysis": analysis.get("nutrient_analysis"),
        "crop_recommendations": crops,
        "fertilizer_recommendation": fertilizer,
        "feature_array": analysis.get("feature_array"),
        "feature_provenance": analysis.get("feature_provenance"),
        "recommendation_confidence": confidence,
        "ai_input_log": analysis.get("ai_input_log"),
        "pipeline": analysis.get("pipeline"),
        "weather_forecast": analysis.get("weather_forecast"),
        "model_stack": analysis.get("model_stack"),
        "retrain_capture": analysis.get("retrain_capture"),
        "recommendations": [
            {
                "category": "Crop recommendation",
                "icon": "plant",
                "data": crops,
                "best_crop": analysis["best_crop"],
                "confidence": confidence.get("score"),
                "confidence_level": confidence.get("level"),
                "predicted_yield": crops[0].get("predicted_yield") if crops else None,
            },
            {
                "category": "Soil nutrients",
                "icon": "science",
                "data": analysis.get("nutrient_analysis"),
            },
            {
                "category": "Irrigation",
                "icon": "water_drop",
                "data": irrigation,
                "status": irrigation.get("status", "unknown"),
                "guidance": irrigation.get("guidance", "Not available"),
            },
            {
                "category": "Disease",
                "icon": "person_with_magnifying_glass",
                "data": disease,
                "health_status": disease.get("health_status", "unknown"),
                "note": disease.get("message"),
            },
            {
                "category": "Fertilizer strategy",
                "icon": "fertilizer_bag",
                "data": fertilizer,
                "strategies": fertilizer.get("recommended_strategies", [])
                if isinstance(fertilizer, dict)
                else [],
            },
            {
                "category": "Weather",
                "icon": "cloud_with_rain",
                "data": analysis["weather_forecast"],
            },
        ],
        "satellite_integration": analysis.get("satellite_integration"),
        "timestamp": analysis["timestamp"],
    }
