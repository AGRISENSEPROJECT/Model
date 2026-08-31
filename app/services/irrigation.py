"""Dynamic irrigation guidance from moisture, crop needs, and weather ET0."""
from __future__ import annotations

from typing import Any

from app.knowledge.irrigation import CROP_WATER_NEEDS
from app.utils.crops import normalize_crop_id


def recommend_irrigation(
    soil_moisture: float,
    crop_type: str,
    temperature: float,
    humidity: float,
    rainfall: float,
    soil_texture: str | None = None,
    *,
    et0_mm: float | None = None,
) -> dict[str, Any]:
    crop_id = normalize_crop_id(crop_type)
    if not crop_id or crop_id not in CROP_WATER_NEEDS:
        return {
            "source": "dynamic_moisture_et",
            "error": f"Crop type {crop_type} not supported for irrigation recommendations",
        }

    crop_needs = CROP_WATER_NEEDS[crop_id]
    optimal = float(crop_needs["optimal_moisture"])
    critical = float(crop_needs["critical_moisture"])
    base_daily = float(crop_needs["daily_water_mm"])

    et0 = float(et0_mm) if et0_mm is not None else max(0.5, 0.15 * temperature + 0.5)
    moisture_deficit = max(0.0, optimal - soil_moisture)
    deficit_ratio = moisture_deficit / max(optimal, 1.0)

    if soil_moisture <= critical:
        urgency = "urgent"
        water_amount = base_daily * (1.0 + deficit_ratio) * (et0 / 4.0)
    elif soil_moisture < optimal:
        urgency = "soon"
        water_amount = base_daily * deficit_ratio * (et0 / 4.0)
    else:
        urgency = "none"
        water_amount = 0.0

    adjustments: list[str] = []
    if rainfall > 10:
        water_amount *= 0.7
        adjustments.append(f"Reduced 30% — recent rainfall {rainfall:.0f} mm")
    if temperature > 35:
        water_amount *= 1.2
        adjustments.append(f"Increased 20% — high air temperature {temperature:.1f}°C")

    if urgency == "none":
        guidance = "Soil moisture is at or above optimal for this crop."
    elif urgency == "urgent":
        guidance = (
            f"Moisture {soil_moisture:.1f}% is at/below critical {critical:.1f}% — "
            "irrigate when feasible."
        )
    else:
        guidance = (
            f"Moisture {soil_moisture:.1f}% is below optimal {optimal:.1f}% — "
            "plan irrigation soon."
        )

    return {
        "source": "dynamic_moisture_et",
        "model_type": "moisture_deficit_x_et0",
        "status": urgency,
        "guidance": guidance,
        "recommended_water_mm": round(float(water_amount), 1),
        "soil_moisture": soil_moisture,
        "optimal_moisture": optimal,
        "critical_moisture": critical,
        "crop_id": crop_id,
        "et0_mm_used": round(et0, 2),
        "moisture_deficit_pct": round(moisture_deficit, 1),
        "weather_adjustment": "; ".join(adjustments) if adjustments else "None",
        "hardcoded_schedule": False,
    }
