"""Rwanda province and district climate / cropping profiles.

Climate: RAB/OFRA AEZ notes (Musanze 1320 mm, Gisenyi/Rubavu 1170 mm),
NISR geography, and published means of ~20 C and ~1000 mm nationally
(Warner et al. 2020). Seasons follow MINAGRI: A Sep-Dec, B Mar-May, C Jun-Aug.
"""

PROVINCE_PROFILES = {
    'Northern': {
        'annual_rainfall_mm': 1320,
        'rainfall_range': (1100, 1550),
        'mean_temp_c': 16.5,
        'summer_max_c': 22.0,
        'winter_min_c': 10.0,
        'humidity': {'A': 78, 'B': 82, 'C': 65},
        'soils': ['loamy', 'clayey', 'sandy'],
        'ph_range': (5.0, 6.4),
        'fertility': (60, 90),
        'altitude_m': (1800, 2500),
        'crops_by_season': {
            'A': ['potato', 'maize', 'beans', 'wheat', 'pea', 'sweet_potato'],
            'B': ['potato', 'beans', 'wheat', 'pea', 'maize', 'tomato'],
            'C': ['potato', 'beans', 'pea', 'tomato', 'sweet_potato'],
        },
        'remarks': 'Volcanic highlands (Musanze, Burera, Nyabihu). Cool, wet, Irish potato and wheat belt.',
    },
    'Western': {
        'annual_rainfall_mm': 1400,
        'rainfall_range': (1150, 1900),
        'mean_temp_c': 17.8,
        'summer_max_c': 24.0,
        'winter_min_c': 12.0,
        'humidity': {'A': 80, 'B': 85, 'C': 68},
        'soils': ['loamy', 'clayey', 'alluvial'],
        'ph_range': (4.6, 6.2),
        'fertility': (50, 85),
        'altitude_m': (1400, 2500),
        'crops_by_season': {
            'A': ['potato', 'tea', 'coffee', 'beans', 'maize', 'banana'],
            'B': ['tea', 'coffee', 'beans', 'potato', 'banana', 'cassava'],
            'C': ['tea', 'coffee', 'beans', 'tomato', 'sweet_potato'],
        },
        'remarks': 'Congo-Nile divide and Lake Kivu. Tea, coffee and highland potato; acid hill soils.',
    },
    'Southern': {
        'annual_rainfall_mm': 1250,
        'rainfall_range': (1050, 1550),
        'mean_temp_c': 19.5,
        'summer_max_c': 26.0,
        'winter_min_c': 14.0,
        'humidity': {'A': 75, 'B': 80, 'C': 62},
        'soils': ['clayey', 'loamy', 'alluvial'],
        'ph_range': (4.8, 6.5),
        'fertility': (45, 80),
        'altitude_m': (1400, 2200),
        'crops_by_season': {
            'A': ['beans', 'maize', 'cassava', 'banana', 'coffee', 'sweet_potato'],
            'B': ['beans', 'cassava', 'banana', 'coffee', 'rice', 'sorghum'],
            'C': ['rice', 'beans', 'tomato', 'soybean', 'sweet_potato'],
        },
        'remarks': 'Acid clays needing lime. Banana-coffee systems; rice in Huye/Gisagara marshlands.',
    },
    'Eastern': {
        'annual_rainfall_mm': 900,
        'rainfall_range': (750, 1100),
        'mean_temp_c': 21.8,
        'summer_max_c': 28.0,
        'winter_min_c': 16.0,
        'humidity': {'A': 62, 'B': 70, 'C': 52},
        'soils': ['sandy', 'loamy', 'clayey', 'alluvial'],
        'ph_range': (5.4, 7.0),
        'fertility': (40, 75),
        'altitude_m': (1200, 1600),
        'crops_by_season': {
            'A': ['maize', 'beans', 'sorghum', 'cassava', 'banana', 'groundnut'],
            'B': ['sorghum', 'cassava', 'banana', 'beans', 'maize', 'rice'],
            'C': ['rice', 'soybean', 'tomato', 'beans', 'sweet_potato'],
        },
        'remarks': 'Drier plateau and Bugesera. Banana, sorghum, cassava, maize; cattle important.',
    },
    'Kigali': {
        'annual_rainfall_mm': 1020,
        'rainfall_range': (900, 1200),
        'mean_temp_c': 20.5,
        'summer_max_c': 27.0,
        'winter_min_c': 15.0,
        'humidity': {'A': 68, 'B': 74, 'C': 55},
        'soils': ['clayey', 'loamy', 'alluvial'],
        'ph_range': (5.2, 6.8),
        'fertility': (45, 78),
        'altitude_m': (1370, 1850),
        'crops_by_season': {
            'A': ['maize', 'beans', 'banana', 'tomato', 'sweet_potato'],
            'B': ['beans', 'maize', 'banana', 'cassava', 'tomato'],
            'C': ['tomato', 'beans', 'rice', 'soybean', 'sweet_potato'],
        },
        'remarks': 'Peri-urban mixed farms and marshland vegetables around Gasabo, Kicukiro, Nyarugenge.',
    },
}

DISTRICT_OFFSETS = {
    'Musanze': {'province': 'Northern', 'rain': 40, 'temp': -1.2},
    'Burera': {'province': 'Northern', 'rain': 20, 'temp': -1.5},
    'Gicumbi': {'province': 'Northern', 'rain': -40, 'temp': -0.4},
    'Rulindo': {'province': 'Northern', 'rain': -80, 'temp': 0.6},
    'Gakenke': {'province': 'Northern', 'rain': -20, 'temp': -0.2},
    'Nyabihu': {'province': 'Western', 'rain': 80, 'temp': -1.0},
    'Rubavu': {'province': 'Western', 'rain': -180, 'temp': 0.4},
    'Rutsiro': {'province': 'Western', 'rain': 120, 'temp': -0.6},
    'Ngororero': {'province': 'Western', 'rain': 40, 'temp': 0.0},
    'Karongi': {'province': 'Western', 'rain': -40, 'temp': 0.8},
    'Nyamasheke': {'province': 'Western', 'rain': 200, 'temp': 0.2},
    'Rusizi': {'province': 'Western', 'rain': 60, 'temp': 2.0},
    'Huye': {'province': 'Southern', 'rain': -30, 'temp': 0.4},
    'Nyanza': {'province': 'Southern', 'rain': -80, 'temp': 0.8},
    'Gisagara': {'province': 'Southern', 'rain': -50, 'temp': 0.6},
    'Nyaruguru': {'province': 'Southern', 'rain': 180, 'temp': -0.8},
    'Nyamagabe': {'province': 'Southern', 'rain': 200, 'temp': -1.0},
    'Ruhango': {'province': 'Southern', 'rain': -60, 'temp': 0.5},
    'Muhanga': {'province': 'Southern', 'rain': -20, 'temp': 0.2},
    'Kamonyi': {'province': 'Southern', 'rain': -70, 'temp': 0.7},
    'Nyagatare': {'province': 'Eastern', 'rain': -80, 'temp': 1.2},
    'Gatsibo': {'province': 'Eastern', 'rain': -40, 'temp': 0.8},
    'Kayonza': {'province': 'Eastern', 'rain': 20, 'temp': 0.4},
    'Rwamagana': {'province': 'Eastern', 'rain': 40, 'temp': 0.2},
    'Ngoma': {'province': 'Eastern', 'rain': 10, 'temp': 0.3},
    'Kirehe': {'province': 'Eastern', 'rain': -20, 'temp': 0.6},
    'Bugesera': {'province': 'Eastern', 'rain': -120, 'temp': 1.5},
    'Gasabo': {'province': 'Kigali', 'rain': 10, 'temp': -0.2},
    'Kicukiro': {'province': 'Kigali', 'rain': -10, 'temp': 0.1},
    'Nyarugenge': {'province': 'Kigali', 'rain': 0, 'temp': 0.2},
}

SEASON_TEMP_OFFSET = {
    'A': 0.5,
    'B': 0.8,
    'C': -1.8,
}

SEASON_RAIN_SHARE = {
    'A': 0.38,
    'B': 0.45,
    'C': 0.17,
}

LAND_USE_TYPES = ['Agricultural', 'Agricultural', 'Agricultural', 'Marshland', 'Homestead', 'Fallow']

# Back-compat alias used by older imports.
DIVISION_PROFILES = PROVINCE_PROFILES
