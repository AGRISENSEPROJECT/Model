DISEASE_PATTERNS = {
    'leaf_blight': {
        'symptoms': ['yellow_spots', 'brown_edges', 'wilting'],
        'conditions': {'humidity_min': 70, 'temp_min': 20, 'temp_max': 30},
        'treatment': 'Apply copper-based fungicide, remove affected leaves'
    },
    'powdery_mildew': {
        'symptoms': ['white_powder', 'leaf_curling'],
        'conditions': {'humidity_min': 60, 'temp_min': 15, 'temp_max': 25},
        'treatment': 'Apply sulfur-based fungicide, improve air circulation'
    },
    'root_rot': {
        'symptoms': ['yellowing_leaves', 'soft_stem', 'wilting'],
        'conditions': {'humidity_min': 80, 'temp_min': 20, 'temp_max': 30},
        'treatment': 'Improve drainage, reduce watering, apply fungicide'
    }
}

CROP_DISEASES = {
    'rice': ['leaf_blight', 'blast', 'sheath_blight'],
    'tomatoes': ['leaf_blight', 'powdery_mildew', 'early_blight'],
    'potatoes': ['late_blight', 'early_blight', 'powdery_mildew'],
    'maize': ['leaf_blight', 'common_rust', 'northern_leaf_blight']
}
