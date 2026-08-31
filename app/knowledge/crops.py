"""Crop suitability ranges (legacy rules + empirical notes from env dataset)."""

CROP_SUITABILITY = {
    "rice": {
        "soil_texture": ["alluvial", "clayey", "loamy"],
        "temperature": {"min": 20, "max": 35},
        "humidity": {"min": 60, "max": 90},
        "rainfall": {"min": 800, "max": 2000},
    },
    "irish_potatoes": {
        "soil_texture": ["loamy", "sandy", "clayey"],
        "temperature": {"min": 15, "max": 25},
        "humidity": {"min": 60, "max": 85},
        "rainfall": {"min": 500, "max": 700},
    },
    "tomatoes": {
        "soil_texture": ["sandy", "loamy", "clayey"],
        "temperature": {"min": 18, "max": 29},
        "humidity": {"min": 50, "max": 80},
        "rainfall": {"min": 400, "max": 600},
    },
    "wheat": {
        "soil_texture": ["loamy", "clayey", "sandy"],
        "temperature": {"min": 15, "max": 25},
        "humidity": {"min": 60, "max": 85},
        "rainfall": {"min": 400, "max": 800},
    },
    "corn": {
        "soil_texture": ["loamy", "clayey", "alluvial"],
        "temperature": {"min": 18, "max": 32},
        "humidity": {"min": 55, "max": 85},
        "rainfall": {"min": 500, "max": 900},
    },
    "barley": {
        "soil_texture": ["loamy", "sandy", "clayey"],
        "temperature": {"min": 12, "max": 24},
        "humidity": {"min": 55, "max": 85},
        "rainfall": {"min": 350, "max": 700},
    },
    "soybean": {
        "soil_texture": ["loamy", "clayey"],
        "temperature": {"min": 18, "max": 30},
        "humidity": {"min": 55, "max": 80},
        "rainfall": {"min": 450, "max": 800},
    },
    "cotton": {
        "soil_texture": ["loamy", "clayey", "alluvial"],
        "temperature": {"min": 20, "max": 35},
        "humidity": {"min": 50, "max": 80},
        "rainfall": {"min": 500, "max": 1000},
    },
    "sugarcane": {
        "soil_texture": ["loamy", "alluvial", "clayey"],
        "temperature": {"min": 20, "max": 35},
        "humidity": {"min": 60, "max": 90},
        "rainfall": {"min": 1000, "max": 2000},
    },
    "sunflower": {
        "soil_texture": ["loamy", "sandy"],
        "temperature": {"min": 18, "max": 30},
        "humidity": {"min": 45, "max": 75},
        "rainfall": {"min": 400, "max": 700},
    },
}
