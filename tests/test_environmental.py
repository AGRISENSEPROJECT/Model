"""Environmental Factors yield-ranking tests."""
from app.services.environmental import (
    recommend_crops_by_yield,
    resolve_dataset_soil,
)


def test_texture_to_dataset_soil():
    assert resolve_dataset_soil("clayey") == "Clay"
    assert resolve_dataset_soil("alluvial") == "Loamy"


def test_yield_ranking_prefers_positive_scores():
    recs = recommend_crops_by_yield(
        soil_texture="loamy",
        temperature=24.0,
        humidity=75.0,
        nitrogen=80.0,
        phosphorus=60.0,
        potassium=45.0,
        soil_ph=6.5,
        wind_speed=10.0,
        top_k=5,
    )
    assert len(recs) >= 3
    assert recs[0]["source"] == "environmental_yield_rf"
    assert "predicted_yield" in recs[0]
    yields = [r["predicted_yield"] for r in recs]
    assert yields == sorted(yields, reverse=True)
