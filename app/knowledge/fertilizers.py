"""Fertilizer lookup tables. Textures peaty/saline fall back to nearest known class."""

_BASE = {
    "rice": {
        "sandy": {"NPK 10-10-10": "Low nitrogen content. Good for initial growth stages."},
        "loamy": {"Urea": "High in nitrogen, good for vegetative growth of rice in loamy soil."},
        "clayey": {"Ammonium Sulfate": "Provides nitrogen and sulfur, good for clayey soils."},
        "alluvial": {"DAP": "High phosphorus content helps in root development in alluvial soils."},
    },
    "irish_potatoes": {
        "sandy": {"NPK 5-10-10": "Low nitrogen, high potassium for tuber development in sandy soil."},
        "loamy": {"NPK 10-20-20": "Balanced for overall growth, with focus on tuber development."},
        "clayey": {"Triple Super Phosphate": "High phosphorus helps with root development in heavy soil."},
        "alluvial": {"MOP": "High potassium content for improving quality and disease resistance."},
    },
    "tomatoes": {
        "sandy": {"NPK 5-10-5": "Balanced nutrients with focus on phosphorus for flowering."},
        "loamy": {"NPK 8-32-16": "High phosphorus promotes flowering and fruiting in tomatoes."},
        "clayey": {"Single Super Phosphate": "Good for breaking down clayey soil and providing phosphorus."},
        "alluvial": {"NPK 12-12-17": "Higher potassium content for fruit development and quality."},
    },
}

_GENERIC_TEXTURES = {
    "sandy": {"NPK 10-10-10": "Balanced starter blend suited to sandy, leach-prone soils."},
    "loamy": {"NPK 15-15-15": "Balanced maintenance blend for productive loamy soils."},
    "clayey": {"NPK 12-12-17": "Slightly higher K to support structure and fruiting on clay."},
    "alluvial": {"NPK 14-14-14": "All-purpose blend for fertile alluvial soils."},
}

_NEW_CROPS = (
    "wheat",
    "corn",
    "barley",
    "soybean",
    "cotton",
    "sugarcane",
    "sunflower",
)

FERTILIZER_RECOMMENDATIONS = dict(_BASE)
for crop in _NEW_CROPS:
    FERTILIZER_RECOMMENDATIONS[crop] = dict(_GENERIC_TEXTURES)

# Env-dataset soils not in original fertilizer table
for crop_recs in FERTILIZER_RECOMMENDATIONS.values():
    crop_recs.setdefault("peaty", crop_recs["loamy"])
    crop_recs.setdefault("saline", crop_recs["sandy"])
