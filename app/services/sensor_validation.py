"""Validate live soil sensor payloads for production field use."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agrisense.sensor")

# Physical probe outputs — pH is NOT measured by the RS485 sensor.
REQUIRED_SENSOR_FIELDS: tuple[str, ...] = (
    "moisture_pct",
    "temperature_c",
    "ec_us_cm",
    "nitrogen",
    "phosphorus",
    "potassium",
)

# Realistic bounds for field probes (adjust after local calibration).
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "moisture_pct": (0.0, 100.0),
    "temperature_c": (-10.0, 60.0),
    "ec_us_cm": (0.0, 20000.0),
    "nitrogen": (0.0, 1999.0),
    "phosphorus": (0.0, 1999.0),
    "potassium": (0.0, 1999.0),
}


@dataclass(frozen=True)
class ValidatedSensorReading:
    moisture_pct: float
    temperature_c: float
    ec_us_cm: float
    nitrogen: float
    phosphorus: float
    potassium: float
    device_id: str | None
    timestamp_ms: int | None
    coordinates: dict[str, float] | None
    farm_id: str | None
    raw_soil: dict[str, Any]


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_sensor_payload(payload: dict[str, Any]) -> ValidatedSensorReading:
    """
    Require all six probe parameters. Reject incomplete, null, or out-of-range values.
    Ignores ph from the device — our probe does not measure pH.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    soil = payload.get("soil")
    if not isinstance(soil, dict):
        raise ValueError("Missing required 'soil' object with six sensor readings")

    parsed: dict[str, float | None] = {}
    missing: list[str] = []
    invalid: list[str] = []

    for field in REQUIRED_SENSOR_FIELDS:
        value = _num(soil.get(field))
        if value is None:
            missing.append(field)
            continue
        lo, hi = FIELD_RANGES[field]
        if value < lo or value > hi:
            invalid.append(f"{field}={value} (allowed {lo}–{hi})")
        parsed[field] = value

    if missing:
        raise ValueError(
            "Incomplete soil reading — all six parameters required: "
            + ", ".join(REQUIRED_SENSOR_FIELDS)
            + f". Missing: {', '.join(missing)}"
        )
    if invalid:
        raise ValueError(
            "Invalid soil reading — out-of-range values: " + "; ".join(invalid)
        )

    if soil.get("ph") is not None:
        logger.warning(
            "Ignoring soil.ph from device %s — pH is not measured by the RS485 probe",
            payload.get("device_id"),
        )

    coordinates = payload.get("coordinates")
    if coordinates is not None and not isinstance(coordinates, dict):
        coordinates = None
    if isinstance(coordinates, dict):
        lat = _num(coordinates.get("lat"))
        lon = _num(coordinates.get("lon"))
        if lat is None or lon is None:
            coordinates = None
        else:
            coordinates = {"lat": lat, "lon": lon}

    farm_id = payload.get("farm_id") or payload.get("location_id")
    ts = payload.get("timestamp_ms")
    timestamp_ms = int(ts) if ts is not None else None

    reading = ValidatedSensorReading(
        moisture_pct=float(parsed["moisture_pct"]),
        temperature_c=float(parsed["temperature_c"]),
        ec_us_cm=float(parsed["ec_us_cm"]),
        nitrogen=float(parsed["nitrogen"]),
        phosphorus=float(parsed["phosphorus"]),
        potassium=float(parsed["potassium"]),
        device_id=payload.get("device_id"),
        timestamp_ms=timestamp_ms,
        coordinates=coordinates,
        farm_id=str(farm_id) if farm_id else None,
        raw_soil=dict(soil),
    )

    logger.info(
        "Validated sensor reading device=%s moisture=%.1f temp=%.1f ec=%.0f N=%.0f P=%.0f K=%.0f",
        reading.device_id,
        reading.moisture_pct,
        reading.temperature_c,
        reading.ec_us_cm,
        reading.nitrogen,
        reading.phosphorus,
        reading.potassium,
    )
    return reading
