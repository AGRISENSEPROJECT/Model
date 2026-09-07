"""Crop-level economic / agronomic reference tables.

Farmgate prices and typical yields live in
data/rwanda/crop_recommendation_factors.csv (see app.knowledge.catalog).
"""

from app.knowledge.catalog import load_catalog

# Days seed -> harvest (typical mid-range varieties)
MATURITY_DAYS = {
    "rice": 120,
    "irish_potatoes": 90,
    "tomatoes": 85,
    "wheat": 120,
    "corn": 110,
    "barley": 100,
    "soybean": 110,
    "cotton": 160,
    "sugarcane": 300,
    "sunflower": 100,
    "beans": 90,
    "cassava": 270,
    "banana": 365,
    "sorghum": 110,
    "sweet_potato": 120,
    "coffee": 365,
    "tea": 365,
    "groundnut": 110,
    "pea": 80,
}

MARKET_PRICE_INDEX = {
    crop_id: rec["farmgate_rwf_kg"] / 1000.0
    for crop_id, rec in load_catalog().items()
}

LABOR_HOURS_HA = {
    "rice": 70,
    "irish_potatoes": 55,
    "tomatoes": 90,
    "wheat": 35,
    "corn": 45,
    "barley": 35,
    "soybean": 40,
    "cotton": 80,
    "sugarcane": 100,
    "sunflower": 40,
    "beans": 50,
    "cassava": 40,
    "banana": 60,
    "sorghum": 35,
    "sweet_potato": 45,
    "coffee": 80,
    "tea": 90,
    "groundnut": 45,
    "pea": 40,
}

BASE_PEST_RISK = {
    "rice": 0.35,
    "irish_potatoes": 0.45,
    "tomatoes": 0.4,
    "wheat": 0.25,
    "corn": 0.35,
    "barley": 0.25,
    "soybean": 0.3,
    "cotton": 0.4,
    "sugarcane": 0.3,
    "sunflower": 0.25,
    "beans": 0.3,
    "cassava": 0.2,
    "banana": 0.25,
    "sorghum": 0.2,
    "sweet_potato": 0.25,
    "coffee": 0.3,
    "tea": 0.25,
    "groundnut": 0.3,
    "pea": 0.25,
}
