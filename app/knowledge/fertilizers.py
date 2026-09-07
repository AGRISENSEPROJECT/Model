"""Fertilizer lookup tables. Textures peaty/saline fall back to nearest known class."""

_BASE = {
    "rice": {
        "sandy": {"NPK 10-10-10": "Low nitrogen content. Good for initial growth stages."},
        "loamy": {"Urea": "High in nitrogen, good for vegetative growth of rice in loamy soil."},
        "clayey": {"Ammonium Sulfate": "Provides nitrogen and sulfur, good for clayey soils."},
        "alluvial": {"DAP": "High phosphorus content helps in root development in alluvial soils."},
    },
    "irish_potatoes": {
        "sandy": {"NPK 17-17-17 + Urea": "RAB/CIP highland potato: NPK at planting, urea at hilling on light volcanic sand."},
        "loamy": {"NPK 17-17-17 + Urea": "Musanze volcanic loam: NPK 17-17-17 plus split urea (RAB potato pack)."},
        "clayey": {"DAP + MOP + Urea": "Heavy clay needs extra K; DAP and MOP at planting."},
        "alluvial": {"NPK 17-17-17": "Valley potato: balanced NPK; keep ridges well drained."},
    },
    "tomatoes": {
        "sandy": {"NPK 5-10-5": "Balanced nutrients with focus on phosphorus for flowering."},
        "loamy": {"NPK 8-32-16": "High phosphorus promotes flowering and fruiting in tomatoes."},
        "clayey": {"Single Super Phosphate": "Good for breaking down clayey soil and providing phosphorus."},
        "alluvial": {"NPK 12-12-17": "Higher potassium content for fruit development and quality."},
    },
}

_GENERIC_TEXTURES = {
    "sandy": {"DAP + Urea": "CIP pack for light Eastern Rwanda soils: DAP at planting, urea topdress."},
    "loamy": {"DAP + Urea": "OFRA Rwanda medium soil: about 41 kg N and 46 kg P2O5/ha for cereals."},
    "clayey": {"DAP + Urea": "Clay holds K; DAP plus urea is the standard CIP blend."},
    "alluvial": {"DAP + Urea + MOP": "Marshland soils may need extra K if residues are removed."},
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
