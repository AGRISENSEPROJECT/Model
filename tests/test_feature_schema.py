from app.features.schema import schema_catalog
from app.features.vector import merge_domains


def test_schema_includes_advisory_history_and_economic():
    cat = schema_catalog()
    assert set(cat["domains"]) == {"soil", "weather", "history", "economic"}
    keys = set(cat["numeric_feature_keys"])
    assert "market_price_per_kg" not in keys
    assert "calcium" not in keys
    assert "soil_ph" in keys
    assert "temperature_c" in keys
    history_keys = {p["key"] for p in cat["domains"]["history"]}
    assert "previous_crop" in history_keys
    assert "season" in history_keys



def test_merge_domains_derives_et0_and_gdd():
    feats = merge_domains(
        soil={"soil_texture": "loamy", "nitrogen": 70, "soil_ph": 6.5},
        weather={"temperature_c": 25, "relative_humidity": 60, "wind_speed": 3},
    )
    assert feats["et0_mm"] > 0
    assert feats["gdd"] > 0
    assert feats["soil_texture"] == "loamy"
