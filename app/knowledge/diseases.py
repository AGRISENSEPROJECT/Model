DISEASE_PATTERNS = {
    "leaf_blight": {
        "symptoms": ["yellow_spots", "brown_edges", "wilting"],
        "conditions": {"humidity_min": 70, "temp_min": 20, "temp_max": 30},
        "treatment": "Apply copper-based fungicide, remove affected leaves",
    },
    "powdery_mildew": {
        "symptoms": ["white_powder", "leaf_curling"],
        "conditions": {"humidity_min": 60, "temp_min": 15, "temp_max": 25},
        "treatment": "Apply sulfur-based fungicide, improve air circulation",
    },
    "root_rot": {
        "symptoms": ["yellowing_leaves", "soft_stem", "wilting"],
        "conditions": {"humidity_min": 80, "temp_min": 20, "temp_max": 30},
        "treatment": "Improve drainage, reduce watering, apply fungicide",
    },
    "late_blight": {
        "symptoms": ["water_soaked_lesions", "white_mold", "tuber_rot"],
        "conditions": {"humidity_min": 80, "temp_min": 10, "temp_max": 20},
        "treatment": "Mancozeb or metalaxyl protectant sprays; destroy volunteer potato plants",
    },
}

CROP_DISEASES = {
    "rice": ["leaf_blight", "blast", "sheath_blight"],
    "tomatoes": ["leaf_blight", "powdery_mildew", "early_blight"],
    "irish_potatoes": ["late_blight", "early_blight", "powdery_mildew"],
    "corn": ["leaf_blight", "root_rot"],
    "wheat": ["powdery_mildew", "root_rot"],
    "soybean": ["root_rot", "powdery_mildew"],
}
