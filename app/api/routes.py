from __future__ import annotations

import base64
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from app.config import ROOT_DIR, UPLOAD_DIR
from app.features.schema import schema_catalog
from app.services.analysis import build_predict_response, run_comprehensive_analysis
from app.services.feedback_store import (
    apply_human_label,
    reject_sample,
    retrain_status,
)
from app.services.sensor_ingest import ingest_sensor_reading
from app.utils.auth import sensor_api_key_error, sensor_api_key_valid
from app.utils.uploads import ensure_upload_dir, save_upload

api_bp = Blueprint("api", __name__)

SAMPLE_BODY = {
    "soil": {
        "soil_texture": "loamy",
        "soil_ph": 6.5,
        "nitrogen": 80,
        "phosphorus": 60,
        "potassium": 45,
        "soil_moisture_vwc": 50,
    },
    "coordinates": {"lat": -1.9441, "lon": 30.0619},
}


def _parse_legacy_nums(source):
    mapping = {
        "temperature": source.get("temperature"),
        "humidity": source.get("humidity"),
        "rainfall": source.get("rainfall"),
        "nitrogen": source.get("nitrogen"),
        "phosphorus": source.get("phosphorus"),
        "potassium": source.get("potassium"),
        "soil_moisture": source.get("soil_moisture", 50),
        "soil_ph": source.get("soil_ph"),
        "wind_speed": source.get("wind_speed"),
    }
    out = {}
    for key, value in mapping.items():
        if value is None or value == "":
            if key == "soil_moisture":
                out[key] = 50.0
            else:
                out[key] = None
            continue
        out[key] = float(value)
    required = ("temperature", "humidity", "rainfall", "nitrogen", "phosphorus", "potassium")
    # Only enforce legacy required fields when caller is on flat legacy form (no nested domains)
    return out, required


def _save_base64_image(data_uri: str) -> str:
    ensure_upload_dir()
    raw = data_uri
    if "," in raw:
        _, raw = raw.split(",", 1)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_json.jpg"
    path.write_bytes(base64.b64decode(raw))
    return str(path)


def _resolve_image_field(image_value: str | None) -> str | None:
    if not image_value:
        return None
    if image_value.startswith("data:image") or (
        len(image_value) > 200 and "/" not in image_value[:20]
    ):
        return _save_base64_image(image_value)
    path = Path(image_value)
    if not path.is_absolute():
        candidate = ROOT_DIR / image_value
        if candidate.exists():
            return str(candidate)
    if path.exists():
        return str(path)
    raise ValueError(f"Image path not found: {image_value}")


def _extract_request_payload():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        soil = data.get("soil")
        weather = data.get("weather")
        history = data.get("history")
        economic = data.get("economic")
        nested = any(isinstance(x, dict) for x in (soil, weather, history, economic))

        crop_type = data.get("crop_type") or None
        if isinstance(crop_type, str) and not crop_type.strip():
            crop_type = None
        coordinates = data.get("coordinates")
        image_path = _resolve_image_field(data.get("image"))
        soil_texture = data.get("soil_texture")
        if isinstance(soil, dict) and soil.get("soil_texture"):
            soil_texture = soil_texture or soil.get("soil_texture")

        legacy, required = _parse_legacy_nums(data)
        if not nested:
            missing = [k for k in required if legacy.get(k) is None]
            if missing and not (soil_texture or image_path or (soil and soil.get("soil_texture"))):
                raise ValueError(
                    "Provide nested {soil,weather,history,economic} OR flat fields "
                    f"(missing: {missing}) with soil_texture/image"
                )
            if not soil_texture and not image_path:
                raise ValueError(
                    "JSON needs soil_texture or image. crop_type is optional."
                )

        return {
            "image_path": image_path,
            "soil_texture": soil_texture,
            "crop_type": crop_type,
            "coordinates": coordinates,
            "soil": soil if isinstance(soil, dict) else None,
            "weather": weather if isinstance(weather, dict) else None,
            "history": history if isinstance(history, dict) else None,
            "economic": economic if isinstance(economic, dict) else None,
            **{k: v for k, v in legacy.items() if v is not None or k == "soil_moisture"},
        }

    if "image" not in request.files:
        raise ValueError("No image uploaded")
    image_path = str(save_upload(request.files["image"]))
    legacy, _ = _parse_legacy_nums(request.form)
    crop_type = request.form.get("crop_type") or None
    soil_texture = request.form.get("soil_texture") or None
    coordinates = None
    lat, lon = request.form.get("lat"), request.form.get("lon")
    if lat and lon:
        coordinates = {"lat": float(lat), "lon": float(lon)}
    return {
        "image_path": image_path,
        "soil_texture": soil_texture,
        "crop_type": crop_type,
        "coordinates": coordinates,
        "soil": None,
        "weather": None,
        "history": None,
        "economic": None,
        **legacy,
    }


def _run_from_request(envelope: str = "predict"):
    payload = _extract_request_payload()
    analysis = run_comprehensive_analysis(**payload)
    if envelope == "raw":
        return analysis
    return build_predict_response(analysis)


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "agrisense"})


@api_bp.get("/feature-schema")
def feature_schema():
    """Exhaustive 4-domain parameter catalog for the precision engine."""
    return jsonify(schema_catalog())


@api_bp.get("/sample-request")
def sample_request():
    return jsonify(
        {
            "endpoint": "POST /predict",
            "content_type": "application/json",
            "note": (
                "Pass soil sensors + coordinates. Weather auto-fills from OpenWeatherMap "
                "when OPENWEATHERMAP_API_KEY is set. crop_type optional."
            ),
            "body": SAMPLE_BODY,
        }
    )


@api_bp.post("/predict")
def predict():
    try:
        return jsonify(_run_from_request("predict"))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("predict failed")
        return jsonify({"error": str(exc)}), 500


@api_bp.post("/comprehensive-analyze")
def comprehensive_analyze():
    try:
        return jsonify(_run_from_request("raw"))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("comprehensive-analyze failed")
        return jsonify({"error": str(exc)}), 500


@api_bp.post("/sensor/reading")
def sensor_reading():
    """
    Receive live soil sensor data from ESP32 devices on the local network.
    Runs full server-side AI analysis and stores the payload + result.
    """
    if not sensor_api_key_valid():
        return jsonify({"error": sensor_api_key_error()}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body required"}), 400

    try:
        current_app.logger.info("Sensor reading received device=%s", data.get("device_id"))
        result = ingest_sensor_reading(data)
        current_app.logger.info(
            "Sensor analysis complete device=%s crop=%s confidence=%s",
            data.get("device_id"),
            result.get("best_crop"),
            (result.get("recommendation_confidence") or {}).get("level"),
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("sensor reading ingest failed")
        return jsonify({"error": str(exc)}), 500


@api_bp.get("/retrain/status")
def get_retrain_status():
    """Inbox / labeled counts for the continuous-improvement pipeline."""
    return jsonify(retrain_status())


@api_bp.post("/retrain/label")
def post_retrain_label():
    """
    Human correction for a captured prediction image.
    Body: {"sample_id": "...", "human_label": "sandy", "notes": "optional"}
    """
    data = request.get_json(silent=True) or {}
    sample_id = data.get("sample_id")
    human_label = data.get("human_label")
    if not sample_id or not human_label:
        return jsonify({"error": "sample_id and human_label are required"}), 400
    try:
        record = apply_human_label(sample_id, human_label, notes=data.get("notes"))
        return jsonify(record)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.post("/retrain/reject")
def post_retrain_reject():
    data = request.get_json(silent=True) or {}
    sample_id = data.get("sample_id")
    if not sample_id:
        return jsonify({"error": "sample_id is required"}), 400
    try:
        return jsonify(reject_sample(sample_id, reason=data.get("reason", "rejected")))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


def register_web_routes(app):
    @app.get("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.get("/playground")
    def playground():
        return send_from_directory(app.template_folder, "playground.html")
