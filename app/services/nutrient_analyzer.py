"""Data-driven nutrient status from sensor readings vs training distributions."""
from __future__ import annotations

from typing import Any

from app.knowledge.environmental_maps import DATASET_CROP_TO_ID
from app.services.dataset_stats import classify_against_distribution, get_dataset_stats
from app.utils.crops import display_crop_name, normalize_crop_id

# EC µS/cm bands — literature ranges; replace with Rwanda field calibration.
EC_BANDS: tuple[tuple[float, float, str], ...] = (
    (0, 400, "low"),
    (400, 800, "normal"),
    (800, 2000, "elevated"),
    (2000, 10000, "high"),
    (10000, 20000, "very_high"),
)

MOISTURE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0, 15, "very_dry"),
    (15, 25, "dry"),
    (25, 45, "moderate"),
    (45, 70, "optimal"),
    (70, 85, "wet"),
    (85, 100, "waterlogged"),
)

TEMP_BANDS: tuple[tuple[float, float, str], ...] = (
    (-10, 10, "cold"),
    (10, 18, "cool"),
    (18, 28, "moderate"),
    (28, 35, "warm"),
    (35, 60, "hot"),
)


def _band(value: float, bands: tuple[tuple[float, float, str], ...]) -> str:
    for lo, hi, label in bands:
        if lo <= value < hi:
            return label
    return bands[-1][2]


def analyze_nutrients(
    *,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ec_us_cm: float,
    moisture_pct: float,
    soil_temperature_c: float,
    crop_id: str | None = None,
) -> dict[str, Any]:
    stats_bundle = get_dataset_stats()
    crop_key = None
    if crop_id:
        # Reverse map canonical id → dataset crop name for per-crop stats
        for name, cid in DATASET_CROP_TO_ID.items():
            if cid == normalize_crop_id(crop_id):
                crop_key = name
                break

    ref = stats_bundle.get("by_crop", {}).get(crop_key or "", {})
    global_ref = stats_bundle.get("global", {})

    def assess(nutrient: str, value: float) -> dict[str, Any]:
        crop_stats = ref.get(nutrient) if ref else None
        base = crop_stats or global_ref.get(nutrient)
        if not base:
            return {
                "value": value,
                "status": "unknown",
                "percentile_estimate": None,
                "reference": "insufficient_training_data",
            }
        label, pct = classify_against_distribution(value, base)
        return {
            "value": value,
            "status": label,
            "percentile_estimate": pct,
            "reference": f"training_csv_{'crop' if crop_stats else 'global'}",
            "distribution": base,
        }

    nutrients = {
        "nitrogen": assess("nitrogen", nitrogen),
        "phosphorus": assess("phosphorus", phosphorus),
        "potassium": assess("potassium", potassium),
    }

    limiting = sorted(
        nutrients.items(),
        key=lambda item: item[1].get("percentile_estimate") or 50.0,
    )
    primary_limit = limiting[0][0] if limiting else None

    return {
        "source": "data_driven_percentiles",
        "dataset_available": stats_bundle.get("available", False),
        "crop_reference": display_crop_name(crop_id) if crop_id else None,
        "nutrients": nutrients,
        "primary_limiting_nutrient": primary_limit,
        "ec": {
            "value_us_cm": ec_us_cm,
            "status": _band(ec_us_cm, EC_BANDS),
            "reference": "literature_bands_pending_local_calibration",
        },
        "moisture": {
            "value_pct": moisture_pct,
            "status": _band(moisture_pct, MOISTURE_BANDS),
        },
        "soil_temperature": {
            "value_c": soil_temperature_c,
            "status": _band(soil_temperature_c, TEMP_BANDS),
        },
        "confidence_note": (
            "Nutrient bands derived from training CSV percentiles; "
            "calibrate against local lab samples for production accuracy."
        ),
    }
