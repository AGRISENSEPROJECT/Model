"""Crop-level economic / agronomic reference tables for precision features."""

# Days seed → harvest (typical mid-range varieties)
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
}

# Relative market price index (unitless; replace with live market API)
MARKET_PRICE_INDEX = {
    "rice": 1.1,
    "irish_potatoes": 0.9,
    "tomatoes": 1.4,
    "wheat": 1.0,
    "corn": 0.95,
    "barley": 0.85,
    "soybean": 1.2,
    "cotton": 1.5,
    "sugarcane": 0.7,
    "sunflower": 1.15,
}

# Labor hours per hectare (relative intensity)
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
}

# Historical pest risk prior (0-1) if field has no log
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
}
