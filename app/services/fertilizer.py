"""Data-driven fertilizer guidance from measured NPK — no static lookup tables."""
from __future__ import annotations

from typing import Any

from app.services.nutrient_analyzer import analyze_nutrients
from app.utils.crops import display_crop_name, normalize_crop_id


def recommend_fertilizer(
    soil_texture: str | None,
    crop_type: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    *,
    ec_us_cm: float | None = None,
    moisture_pct: float | None = None,
    soil_temperature_c: float | None = None,
) -> dict[str, Any]:
    """
    Dynamic nutrient management advice from sensor readings vs training distributions.
    Does NOT prescribe exact kg/ha without validated agronomic models.
    """
    crop_id = normalize_crop_id(crop_type)
    if not crop_id:
        return {
            "source": "data_driven",
            "error": f"Unsupported crop_type: {crop_type}",
        }

    analysis = analyze_nutrients(
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ec_us_cm=float(ec_us_cm or 0),
        moisture_pct=float(moisture_pct or 0),
        soil_temperature_c=float(soil_temperature_c or 0),
        crop_id=crop_id,
    )

    nutrients = analysis["nutrients"]
    n_status = nutrients["nitrogen"]["status"]
    p_status = nutrients["phosphorus"]["status"]
    k_status = nutrients["potassium"]["status"]

    strategies: list[str] = []
    primary = analysis.get("primary_limiting_nutrient")
    if primary:
        label = primary[0].upper()
        status = nutrients[primary]["status"]
        if status in ("very_low", "low"):
            strategies.append(
                f"{label} appears below typical levels for {display_crop_name(crop_id)} "
                f"— consider soil testing before applying fertilizer."
            )
        elif status == "very_high":
            strategies.append(
                f"{label} appears above typical levels — avoid excess {label} applications."
            )

    if analysis["ec"]["status"] in ("high", "very_high"):
        strategies.append(
            "Elevated EC may indicate salinity stress — leaching or gypsum may be needed; "
            "confirm with laboratory EC measurement."
        )

    return {
        "source": "data_driven_npk_analysis",
        "model_type": "percentile_vs_training_csv",
        "crop_id": crop_id,
        "crop": display_crop_name(crop_id),
        "soil_texture": soil_texture or "unknown",
        "soil_npk_status": f"N: {n_status}, P: {p_status}, K: {k_status}",
        "nutrient_analysis": analysis,
        "primary_limiting_nutrient": primary,
        "recommended_strategies": strategies,
        "disclaimer": (
            "No exact fertilizer rates provided — rates require local calibration "
            "and laboratory validation."
        ),
        "hardcoded_lookup": False,
    }
