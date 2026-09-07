"""Multi-factor ranker: soil, season, rotation, yield, and farmgate income."""
from app.knowledge.catalog import load_catalog
from app.knowledge.seasons import current_season, infer_province, season_from_month
from app.services.crop_ranker import rank_crops
from app.utils.crops import normalize_crop_id


def test_catalog_includes_rwanda_staples():
    catalog = load_catalog()
    for crop_id in ("corn", "beans", "irish_potatoes", "cassava", "banana", "rice"):
        assert crop_id in catalog
        assert catalog[crop_id]["farmgate_rwf_kg"] > 0
        assert catalog[crop_id]["typical_yield_t_ha"] > 0


def test_aliases_include_rwanda_crops():
    assert normalize_crop_id("maize") == "corn"
    assert normalize_crop_id("beans") == "beans"
    assert normalize_crop_id("cassava") == "cassava"


def test_season_a_in_september():
    assert season_from_month(9) == "A"
    assert current_season() in {"A", "B", "C"}


def test_kigali_coordinates_map_to_province():
    assert infer_province(-1.9441, 30.0619) == "Kigali"


def test_ranker_returns_income_and_rotation():
    result = rank_crops(
        conditions={
            "soil_texture": "loamy",
            "soil_ph": 6.2,
            "nitrogen": 80,
            "phosphorus": 55,
            "potassium": 50,
            "temperature_c": 19,
            "precip_accum_mm": 900,
            "relative_humidity": 72,
            "province": "Northern",
        },
        history={"previous_crop": "maize", "season": "A"},
        economic={"maximize_income": True},
        ml_rows=[{"crop_id": "corn", "predicted_yield": 4.0}],
        top_k=8,
    )
    recs = result["crop_recommendations"]
    assert len(recs) >= 5
    top = recs[0]
    assert "expected_income_rwf_ha" in top
    assert "farmgate_price_rwf_kg" in top
    assert "agronomic_fit" in top
    assert "factor_breakdown" in top
    assert result["season"] == "A"
    assert result["previous_crop"] == "corn"
    assert result["income_maximizing_crop"]
    assert isinstance(result["crop_rotation_plan"], list)
    ids = {r["crop_id"] for r in recs}
    assert "beans" in ids or any(r["family"] == "legume" for r in recs)


def test_price_override_changes_income():
    base = rank_crops(
        conditions={"soil_texture": "loamy", "temperature_c": 22, "precip_accum_mm": 800},
        economic={"maximize_income": True},
        top_k=20,
    )
    expensive_beans = rank_crops(
        conditions={"soil_texture": "loamy", "temperature_c": 22, "precip_accum_mm": 800},
        economic={"maximize_income": True, "market_prices": {"beans": 5000}},
        top_k=20,
    )
    beans_base = next(r for r in base["crop_recommendations"] if r["crop_id"] == "beans")
    beans_hi = next(r for r in expensive_beans["crop_recommendations"] if r["crop_id"] == "beans")
    assert beans_hi["expected_income_rwf_ha"] > beans_base["expected_income_rwf_ha"]
