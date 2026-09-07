"""
Lean feature schema: sensor + app + OpenWeatherMap only.

Sensor probe (RS485) measures: temperature, moisture, EC, N, P, K — NOT pH.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["live", "derived", "weather_api", "missing", "lab_only"]
Domain = Literal["soil", "weather", "history", "economic"]


@dataclass(frozen=True)
class ParamSpec:
    key: str
    domain: Domain
    dtype: str
    unit: str
    description: str
    default: float | int | str | None
    status: Status


SOIL_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "soil_temperature_c",
        "soil",
        "float",
        "°C",
        "Soil temperature from RS485 probe",
        None,
        "live",
    ),
    ParamSpec(
        "soil_moisture_vwc",
        "soil",
        "float",
        "%",
        "Volumetric moisture from sensor",
        None,
        "live",
    ),
    ParamSpec(
        "ec_us_cm",
        "soil",
        "float",
        "µS/cm",
        "Electrical conductivity from sensor",
        None,
        "live",
    ),
    ParamSpec(
        "nitrogen",
        "soil",
        "float",
        "ppm",
        "Nitrogen (N) from sensor",
        None,
        "live",
    ),
    ParamSpec(
        "phosphorus",
        "soil",
        "float",
        "ppm",
        "Phosphorus (P) from sensor",
        None,
        "live",
    ),
    ParamSpec(
        "potassium",
        "soil",
        "float",
        "ppm",
        "Potassium (K) from sensor",
        None,
        "live",
    ),
    ParamSpec(
        "soil_ph",
        "soil",
        "float",
        "pH",
        "Lab analysis or future pH sensor — NOT from current RS485 probe",
        None,
        "lab_only",
    ),
    ParamSpec(
        "soil_texture",
        "soil",
        "str",
        "class",
        "Texture from CNN image or farmer/lab input",
        None,
        "live",
    ),
)

WEATHER_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "temperature_c",
        "weather",
        "float",
        "°C",
        "Air temperature (OpenWeatherMap)",
        None,
        "weather_api",
    ),
    ParamSpec("temp_max_c", "weather", "float", "°C", "Forecast max temperature", None, "derived"),
    ParamSpec("temp_min_c", "weather", "float", "°C", "Forecast min temperature", None, "derived"),
    ParamSpec(
        "relative_humidity",
        "weather",
        "float",
        "%",
        "Relative humidity (OpenWeatherMap)",
        None,
        "weather_api",
    ),
    ParamSpec(
        "precip_accum_mm",
        "weather",
        "float",
        "mm",
        "Forecast / lifecycle precip",
        None,
        "weather_api",
    ),
    ParamSpec(
        "wind_speed",
        "weather",
        "float",
        "m/s",
        "Wind speed (OpenWeatherMap)",
        None,
        "weather_api",
    ),
    ParamSpec(
        "gdd",
        "weather",
        "float",
        "°C·day",
        "Growing degree days from forecast",
        None,
        "derived",
    ),
    ParamSpec("et0_mm", "weather", "float", "mm/day", "ET0 proxy from weather", None, "derived"),
)

HISTORY_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "previous_crop",
        "history",
        "str",
        "crop_id",
        "Last harvested crop — used for rotation scoring",
        None,
        "live",
    ),
    ParamSpec(
        "season",
        "history",
        "str",
        "A|B|C",
        "Rwanda MINAGRI season (inferred from date if omitted)",
        None,
        "derived",
    ),
    ParamSpec(
        "province",
        "history",
        "str",
        "name",
        "Province (inferred from GPS if omitted)",
        None,
        "derived",
    ),
)

ECONOMIC_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "maximize_income",
        "economic",
        "bool",
        "flag",
        "Weight farmgate income higher than raw yield",
        1,
        "live",
    ),
    ParamSpec(
        "market_prices",
        "economic",
        "object",
        "RWF/kg",
        "Optional farmgate price overrides per crop_id",
        None,
        "live",
    ),
)

ALL_PARAMS: tuple[ParamSpec, ...] = SOIL_PARAMS + WEATHER_PARAMS
ADVISORY_PARAMS: tuple[ParamSpec, ...] = HISTORY_PARAMS + ECONOMIC_PARAMS

# Legacy model artifact uses soil_ph + no EC/soil_temperature — kept for backward compat.
LEGACY_NUMERIC_FEATURE_KEYS: tuple[str, ...] = (
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "soil_moisture_vwc",
    "precip_accum_mm",
    "temperature_c",
    "temp_max_c",
    "temp_min_c",
    "relative_humidity",
    "wind_speed",
    "gdd",
    "et0_mm",
)

NUMERIC_FEATURE_KEYS: tuple[str, ...] = LEGACY_NUMERIC_FEATURE_KEYS

CATEGORICAL_FEATURE_KEYS: tuple[str, ...] = ("soil_texture", "crop_id")

SENSOR_LIVE_KEYS: tuple[str, ...] = (
    "soil_temperature_c",
    "soil_moisture_vwc",
    "ec_us_cm",
    "nitrogen",
    "phosphorus",
    "potassium",
)


def schema_catalog() -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {
        "soil": [],
        "weather": [],
        "history": [],
        "economic": [],
    }
    for p in ALL_PARAMS + ADVISORY_PARAMS:
        by_domain[p.domain].append(asdict(p))
    return {
        "domains": by_domain,
        "numeric_feature_keys": list(NUMERIC_FEATURE_KEYS),
        "categorical_feature_keys": list(CATEGORICAL_FEATURE_KEYS),
        "required_sensor_fields": [
            "temperature_c",
            "moisture_pct",
            "ec_us_cm",
            "nitrogen",
            "phosphorus",
            "potassium",
        ],
        "sensor_note": "Probe measures 6 params; pH requires lab or separate sensor",
        "ranking_note": (
            "ML yield uses soil+weather only. Season, rotation, and farmgate prices "
            "are applied by the multi-factor ranker and do not enter the GBR matrix."
        ),
        "counts": {d: len(v) for d, v in by_domain.items()},
    }
