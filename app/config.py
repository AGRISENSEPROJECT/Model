"""Central configuration for Agrisense production backend."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("AGRISENSE_ARTIFACTS", ROOT_DIR / "artifacts"))
DATA_DIR = Path(os.environ.get("AGRISENSE_DATA", ROOT_DIR / "data"))
UPLOAD_DIR = Path(os.environ.get("AGRISENSE_UPLOADS", ROOT_DIR / "uploads"))

SOIL_MODEL_PATH = ARTIFACTS_DIR / "soil_texture_mobilenetv2.keras"
CROP_MODEL_PATH = ARTIFACTS_DIR / "crop_predictor.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
SOIL_CLASS_INDICES_PATH = ARTIFACTS_DIR / "soil_class_indices.json"
YIELD_MODEL_PATH = ARTIFACTS_DIR / "yield_predictor.pkl"
QUALITY_MODEL_PATH = ARTIFACTS_DIR / "soil_quality_predictor.pkl"
ENV_META_PATH = ARTIFACTS_DIR / "environmental_model_meta.json"
ENV_DATASET_PATH = DATA_DIR / "environmental" / "crop_yield_dataset.csv"

# Keras flow_from_directory sorts class folders alphabetically.
# This is the ground-truth index map when training without custom class_mode.
DEFAULT_SOIL_CLASS_INDICES = {
    "alluvial": 0,
    "clayey": 1,
    "loamy": 2,
    "sandy": 3,
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Canonical crop IDs (aligned with Environmental Factors dataset)
CANONICAL_CROPS = (
    "rice",
    "irish_potatoes",
    "tomatoes",
    "wheat",
    "corn",
    "barley",
    "soybean",
    "cotton",
    "sugarcane",
    "sunflower",
)

CROP_DISPLAY_NAMES = {
    "rice": "Rice",
    "irish_potatoes": "Irish Potatoes",
    "tomatoes": "Tomatoes",
    "wheat": "Wheat",
    "corn": "Maize",
    "barley": "Barley",
    "soybean": "Soybean",
    "cotton": "Cotton",
    "sugarcane": "Sugarcane",
    "sunflower": "Sunflower",
    "beans": "Beans",
    "cassava": "Cassava",
    "banana": "Banana",
    "sorghum": "Sorghum",
    "sweet_potato": "Sweet Potato",
    "coffee": "Coffee",
    "tea": "Tea",
    "groundnut": "Groundnut",
    "pea": "Pea",
}

# Legacy aliases from older API / UI payloads
CROP_ALIASES = {
    "rice": "rice",
    "irish potatoes": "irish_potatoes",
    "irish_potatoes": "irish_potatoes",
    "potatoes": "irish_potatoes",
    "potato": "irish_potatoes",
    "tomatoes": "tomatoes",
    "tomato": "tomatoes",
    "wheat": "wheat",
    "corn": "corn",
    "maize": "corn",
    "barley": "barley",
    "soybean": "soybean",
    "soy": "soybean",
    "cotton": "cotton",
    "sugarcane": "sugarcane",
    "sunflower": "sunflower",
    "beans": "beans",
    "bean": "beans",
    "cassava": "cassava",
    "banana": "banana",
    "sorghum": "sorghum",
    "sweet potato": "sweet_potato",
    "sweet_potato": "sweet_potato",
    "coffee": "coffee",
    "tea": "tea",
    "groundnut": "groundnut",
    "groundnuts": "groundnut",
    "pea": "pea",
    "peas": "pea",
}


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "agrisense-dev-change-me")
    SENSOR_API_KEY = os.environ.get("SENSOR_API_KEY", "agrisense-local")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    JSON_SORT_KEYS = False
    SWAGGER = {"title": "Agrisense API", "uiversion": 3}
