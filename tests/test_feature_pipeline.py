"""Legacy RF feature pipeline matches training (label-encode + scale)."""
import numpy as np

from app.config import CANONICAL_CROPS
from app.services.crop_recommender import build_feature_vector, recommend_crops_ml


def test_feature_vector_shape():
    vec = build_feature_vector("loamy", 22.0, 65.0, 550.0)
    assert vec.shape == (1, 4)


def test_ml_recommendation_returns_ranked_crops():
    recs = recommend_crops_ml("clayey", 28.0, 75.0, 1200.0)
    assert len(recs) >= 1
    assert recs[0]["source"] == "random_forest"
    assert "probability" in recs[0]
    assert recs[0]["crop_id"] in set(CANONICAL_CROPS)
    probs = [r["probability"] for r in recs]
    assert probs == sorted(probs, reverse=True)


def test_wet_heavy_soil_prefers_water_loving_crop():
    """High rainfall + alluvial should rank a high-water crop near the top."""
    recs = recommend_crops_ml("alluvial", 28.0, 80.0, 1400.0)
    top = {r["crop_id"] for r in recs[:3]}
    assert top & {"rice", "sugarcane", "corn"}
