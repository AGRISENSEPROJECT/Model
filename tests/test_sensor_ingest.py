"""Tests for device sensor ingestion."""
from __future__ import annotations

COMPLETE_SOIL = {
    "moisture_pct": 42.5,
    "temperature_c": 24.0,
    "ec_us_cm": 650,
    "nitrogen": 80,
    "phosphorus": 55,
    "potassium": 40,
}


def test_sensor_reading_requires_all_six_fields(client, monkeypatch):
    monkeypatch.setenv("SENSOR_API_KEY", "")
    response = client.post("/api/sensor/reading", json={"device_id": "test"})
    assert response.status_code == 400
    err = response.get_json()["error"].lower()
    assert "soil" in err or "required" in err or "six" in err


def test_sensor_reading_rejects_partial_payload(client, monkeypatch):
    monkeypatch.setenv("SENSOR_API_KEY", "")
    partial = dict(COMPLETE_SOIL)
    del partial["ec_us_cm"]
    response = client.post(
        "/api/sensor/reading",
        json={"device_id": "test", "soil": partial},
    )
    assert response.status_code == 400
    assert "ec_us_cm" in response.get_json()["error"] or "six" in response.get_json()["error"].lower()


def test_sensor_reading_accepts_complete_payload(client, monkeypatch):
    monkeypatch.setenv("SENSOR_API_KEY", "")

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "best_crop": "Maize",
            "best_crop_id": "corn",
            "soil_analysis": {"texture": "unknown"},
            "crop_recommendations": [{"crop": "Maize", "suitability_score": 88, "crop_id": "corn"}],
            "irrigation_recommendation": {},
            "disease_analysis": {},
            "fertilizer_recommendation": {},
            "weather_forecast": {},
            "recommendation_confidence": {"level": "medium", "score": 55},
            "ai_input_log": {},
            "pipeline": [],
            "timestamp": "2026-01-01T00:00:00",
        }

    monkeypatch.setattr("app.services.sensor_ingest.run_comprehensive_analysis", fake_run)
    monkeypatch.setattr(
        "app.services.sensor_ingest.build_predict_response",
        lambda analysis: {
            "soil_texture": "unknown",
            "crop_recommendations": analysis["crop_recommendations"],
            "recommendations": [],
        },
    )
    monkeypatch.setattr(
        "app.services.sensor_ingest.save_device_reading",
        lambda payload, analysis: "memory://test.json",
    )
    monkeypatch.setattr(
        "app.services.sensor_ingest.save_field_visit",
        lambda **kwargs: "memory://visit.json",
    )

    response = client.post(
        "/api/sensor/reading",
        json={
            "device_id": "esp32-test",
            "soil": COMPLETE_SOIL,
            "coordinates": {"lat": -1.9441, "lon": 30.0619},
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["device_id"] == "esp32-test"
    assert captured["nitrogen"] == 80.0
    assert captured["production_mode"] is True
