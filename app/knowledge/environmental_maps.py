"""Mappings between Agrisense IDs and Environmental Factors dataset labels."""
from __future__ import annotations

# Dataset Soil_Type → canonical texture used by CNN / API
DATASET_SOIL_TO_TEXTURE = {
    "Clay": "clayey",
    "Loamy": "loamy",
    "Sandy": "sandy",
    "Peaty": "peaty",
    "Saline": "saline",
}

# CNN / API texture → closest dataset Soil_Type for yield inference
TEXTURE_TO_DATASET_SOIL = {
    "clayey": "Clay",
    "loamy": "Loamy",
    "sandy": "Sandy",
    "alluvial": "Loamy",  # fertile river soils ≈ loamy productivity in this set
    "peaty": "Peaty",
    "saline": "Saline",
}

# Dataset Crop_Type → canonical crop_id
DATASET_CROP_TO_ID = {
    "Rice": "rice",
    "Tomato": "tomatoes",
    "Potato": "irish_potatoes",
    "Wheat": "wheat",
    "Corn": "corn",
    "Barley": "barley",
    "Soybean": "soybean",
    "Cotton": "cotton",
    "Sugarcane": "sugarcane",
    "Sunflower": "sunflower",
}

ID_TO_DATASET_CROP = {v: k for k, v in DATASET_CROP_TO_ID.items()}

# Default pH when user does not supply one (dataset mode per soil)
DEFAULT_PH_BY_SOIL = {
    "Clay": 6.25,
    "Loamy": 6.5,
    "Sandy": 6.75,
    "Peaty": 5.5,
    "Saline": 8.0,
}

DEFAULT_WIND_SPEED = 10.0
