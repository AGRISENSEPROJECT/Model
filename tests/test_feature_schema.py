from app.features.schema import schema_catalog
from app.features.vector import merge_domains


def test_schema_is_soil_and_weather_only():
    cat = schema_catalog()
    assert set(cat["domains"]) == {"soil", "weather"}
    assert "history" not in cat["domains"]
    assert "economic" not in cat["domains"]
    keys = set(cat["numeric_feature_keys"])
    assert "market_price_per_kg" not in keys
    assert "calcium" not in keys
    assert "soil_ph" in keys
    assert "temperature_c" in keys


def test_merge_domains_derives_et0_and_gdd():
    feats = merge_domains(
        soil={"soil_texture": "loamy", "nitrogen": 70, "soil_ph": 6.5},
        weather={"temperature_c": 25, "relative_humidity": 60, "wind_speed": 3},
    )
    assert feats["et0_mm"] > 0
    assert feats["gdd"] > 0
    assert feats["soil_texture"] == "loamy"
