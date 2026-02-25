CROP_WATER_NEEDS = {
    'rice': {
        'daily_water_mm': 10,
        'optimal_moisture': 60,
        'critical_moisture': 35,
        'growth_stages': {
            'vegetative': {'multiplier': 0.8, 'duration_days': 30},
            'reproductive': {'multiplier': 1.2, 'duration_days': 25},
            'ripening': {'multiplier': 0.6, 'duration_days': 20}
        }
    },
    'tomatoes': {
        'daily_water_mm': 6,
        'optimal_moisture': 55,
        'critical_moisture': 30,
        'growth_stages': {
            'vegetative': {'multiplier': 0.7, 'duration_days': 25},
            'flowering': {'multiplier': 1.0, 'duration_days': 20},
            'fruiting': {'multiplier': 1.3, 'duration_days': 30}
        }
    },
    'maize': {
        'daily_water_mm': 8,
        'optimal_moisture': 50,
        'critical_moisture': 25,
        'growth_stages': {
            'establishment': {'multiplier': 0.6, 'duration_days': 15},
            'vegetative': {'multiplier': 1.0, 'duration_days': 30},
            'reproductive': {'multiplier': 1.2, 'duration_days': 25}
        }
    }
}

SOIL_MOISTURE_THRESHOLDS = {
    'sandy': {'field_capacity': 15, 'wilting_point': 5},
    'loamy': {'field_capacity': 35, 'wilting_point': 10},
    'clayey': {'field_capacity': 45, 'wilting_point': 20},
    'alluvial': {'field_capacity': 40, 'wilting_point': 15}
}
