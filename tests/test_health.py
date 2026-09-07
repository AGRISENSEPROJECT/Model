"""Production health, confidence gates, and weather fallback (no TensorFlow)."""
from __future__ import annotations

from app.services.confidence import compute_recommendation_confidence
from app.services.production_ready import production_status, soil_retrain_gate
from app.services.weather import fetch_open_meteo, fetch_weather_bundle
from app.utils.crops import display_crop_name


STUB_ANALYSIS = {
    "soil_analysis": {"texture": "loamy"},
    "nutrient_analysis": None,
    "crop_recommendations": [
        {"crop_id": "corn", "suitability_score": 80, "predicted_yield": 4.1}
    ],
    "best_crop": "Maize",
    "irrigation_recommendation": {},
    "disease_analysis": {},
    "fertilizer_recommendation": {},
    "weather_forecast": {},
    "recommendation_confidence": {"score": 50, "level": "medium"},
    "timestamp": "2026-01-01T00:00:00",
}


def test_display_maize_not_corn():
    assert display_crop_name("corn") == "Maize"


def test_health_reports_production_gates(client):
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    body = response.get_json()
    assert body["service"] == "agrisense"
    assert "production_ready" in body
    assert "sandy_recall" in body["soil_cnn"]
    assert body["weather"]["open_meteo_fallback"] is True
    assert set(body["dataset"]["train"]) >= {"alluvial", "clayey", "loamy", "sandy"}


def test_soil_retrain_gate_rejects_low_sandy_recall():
    rejected = soil_retrain_gate(
        {"accuracy": 0.88, "report": {"sandy": {"recall": 0.41}}}
    )
    assert rejected["ok"] is False
    assert any("sandy_recall" in r for r in rejected["reasons"])

    accepted = soil_retrain_gate(
        {"accuracy": 0.88, "report": {"sandy": {"recall": 0.62}}}
    )
    assert accepted["ok"] is True


def test_low_cnn_confidence_penalizes_score():
    high = compute_recommendation_confidence(
        feature_provenance={
            "nitrogen": "live",
            "phosphorus": "live",
            "potassium": "live",
            "soil_moisture_vwc": "live",
            "ec_us_cm": "live",
            "soil_temperature_c": "live",
            "temperature_c": "live",
            "relative_humidity": "live",
        },
        has_weather_api=True,
        has_soil_texture_cnn=True,
        soil_texture_source="cnn",
        model_name="precision_ml_yield_ranker",
        soil_texture_confidence=0.91,
        soil_texture_label="loamy",
    )
    low = compute_recommendation_confidence(
        feature_provenance={
            "nitrogen": "live",
            "phosphorus": "live",
            "potassium": "live",
            "soil_moisture_vwc": "live",
            "ec_us_cm": "live",
            "soil_temperature_c": "live",
            "temperature_c": "live",
            "relative_humidity": "live",
        },
        has_weather_api=True,
        has_soil_texture_cnn=True,
        soil_texture_source="cnn",
        model_name="precision_ml_yield_ranker",
        soil_texture_confidence=0.40,
        soil_texture_label="sandy",
    )
    assert high["score"] > low["score"]
    assert any("production floor" in f for f in low["factors"])
    assert any("Sandy" in f for f in low["factors"])


def test_predict_defaults_to_production_mode(client, monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return STUB_ANALYSIS

    monkeypatch.setattr("app.api.routes.run_comprehensive_analysis", fake_run)
    response = client.post(
        "/api/predict",
        json={
            "soil": {"soil_texture": "loamy", "soil_ph": 6.5, "nitrogen": 80},
            "coordinates": {"lat": -1.9441, "lon": 30.0619},
        },
    )
    assert response.status_code == 200
    assert captured["production_mode"] is True


def test_predict_debug_disables_production_mode(client, monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return STUB_ANALYSIS

    monkeypatch.setattr("app.api.routes.run_comprehensive_analysis", fake_run)
    response = client.post(
        "/api/predict?debug=true",
        json={"soil": {"soil_texture": "loamy", "nitrogen": 80}},
    )
    assert response.status_code == 200
    assert captured["production_mode"] is False


def test_open_meteo_maps_rwanda_forecast(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "timezone": "Africa/Kigali",
                "current": {
                    "temperature_2m": 21.5,
                    "relative_humidity_2m": 70,
                    "precipitation": 1.2,
                    "wind_speed_10m": 2.0,
                },
                "daily": {
                    "temperature_2m_max": [24.0, 25.0],
                    "temperature_2m_min": [14.0, 15.0],
                    "precipitation_sum": [3.0, 1.0],
                },
            }

    monkeypatch.setattr("app.services.weather.requests.get", lambda *a, **k: FakeResp())
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
    monkeypatch.delenv("OWM_API_KEY", raising=False)
    bundle = fetch_weather_bundle(-1.9441, 30.0619)
    assert bundle["provider"] == "open-meteo"
    assert bundle["weather"]["temperature_c"] == 21.5
    assert bundle["weather"]["precip_accum_mm"] == 4.0
    assert fetch_open_meteo(-1.94, 30.06)["openweathermap"]["timezone"] == "Africa/Kigali"


def test_production_status_shape():
    status = production_status()
    assert status["weather"]["open_meteo_fallback"] is True
    assert "precision_ranker" in status["artifacts"]
