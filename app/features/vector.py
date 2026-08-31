"""Assemble lean sensor + weather FeatureArray."""
from __future__ import annotations

from typing import Any

from app.features.schema import ALL_PARAMS, CATEGORICAL_FEATURE_KEYS, NUMERIC_FEATURE_KEYS
from app.utils.crops import normalize_crop_id

# Weather-only defaults when API unavailable (never applied to sensor keys in production).
_WEATHER_DEFAULTS: dict[str, Any] = {
    "temperature_c": 24.0,
    "temp_max_c": 30.0,
    "temp_min_c": 18.0,
    "relative_humidity": 70.0,
    "precip_accum_mm": 800.0,
    "wind_speed": 3.0,
    "gdd": 900.0,
    "et0_mm": 4.5,
}

_SOIL_KEYS = {p.key for p in ALL_PARAMS if p.domain == "soil"}
_WEATHER_KEYS = {p.key for p in ALL_PARAMS if p.domain == "weather"}


def estimate_gdd(temp_mean: float, temp_max: float, temp_min: float, days: float = 5.0) -> float:
    daily = max(0.0, ((temp_max + temp_min) / 2.0) - 10.0)
    if daily == 0 and temp_mean:
        daily = max(0.0, temp_mean - 10.0)
    return round(daily * days, 2)


def estimate_et0(temp_c: float, humidity: float, wind_speed: float) -> float:
    vapor_deficit = max(0.0, (100.0 - humidity) / 100.0)
    return round(max(0.5, 0.15 * temp_c + 0.8 * vapor_deficit * max(wind_speed, 0.1) + 0.5), 3)


def merge_domains(
    soil: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    flat: dict[str, Any] | None = None,
    *,
    production_mode: bool = False,
    allow_weather_defaults: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """
    Merge soil + weather features.

    production_mode=True: never invent sensor values; weather defaults only if
    allow_weather_defaults=True (reduces confidence downstream).
    """
    out: dict[str, Any] = {p.key: None for p in ALL_PARAMS}
    flat = dict(flat or {})

    aliases = {
        "temperature": "temperature_c",
        "humidity": "relative_humidity",
        "rainfall": "precip_accum_mm",
        "soil_moisture": "soil_moisture_vwc",
        "n": "nitrogen",
        "p": "phosphorus",
        "k": "potassium",
        "ec": "ec_us_cm",
    }
    for src, dst in aliases.items():
        if src in flat and flat[src] is not None and dst not in flat:
            flat[dst] = flat[src]

    allowed = {p.key for p in ALL_PARAMS} | set(CATEGORICAL_FEATURE_KEYS)
    for block in (soil, weather, flat):
        if not block:
            continue
        for key, value in block.items():
            if value is None or value == "":
                continue
            if key in allowed:
                out[key] = value

    if not production_mode:
        for key, default in _WEATHER_DEFAULTS.items():
            if out.get(key) is None:
                out[key] = default
        for key in _SOIL_KEYS:
            if out.get(key) is None:
                spec = next(p for p in ALL_PARAMS if p.key == key)
                if spec.default is not None:
                    out[key] = spec.default
        if out.get("soil_texture") is None:
            out["soil_texture"] = "loamy"
    else:
        if allow_weather_defaults:
            for key, default in _WEATHER_DEFAULTS.items():
                if key in _WEATHER_KEYS and out.get(key) is None:
                    out[key] = default

    # Legacy yield model expects temperature_c + soil_ph — map probe temp for air if missing.
    if out.get("temperature_c") is None and out.get("soil_temperature_c") is not None:
        out["temperature_c"] = out["soil_temperature_c"]

    if out.get("temp_max_c") is None and out.get("temperature_c") is not None:
        t = float(out["temperature_c"])
        out["temp_max_c"] = t + 5
        out["temp_min_c"] = t - 5

    t = out.get("temperature_c")
    if t is not None:
        tmax = float(out.get("temp_max_c") or t + 5)
        tmin = float(out.get("temp_min_c") or t - 5)
        out["temp_max_c"] = tmax
        out["temp_min_c"] = tmin
        if out.get("gdd") is None:
            out["gdd"] = estimate_gdd(float(t), tmax, tmin, days=5.0)
        if out.get("et0_mm") is None:
            rh = float(out.get("relative_humidity") or _WEATHER_DEFAULTS["relative_humidity"])
            ws = float(out.get("wind_speed") or _WEATHER_DEFAULTS["wind_speed"])
            out["et0_mm"] = estimate_et0(float(t), rh, ws)

    if out.get("soil_texture"):
        out["soil_texture"] = str(out["soil_texture"]).strip().lower()

    return out


def enrich_for_crop(features: dict[str, Any], crop_id: str) -> dict[str, Any]:
    row = dict(features)
    row["crop_id"] = normalize_crop_id(crop_id) or crop_id
    return row


def provenance_report(features: dict[str, Any], provided_keys: set[str]) -> dict[str, str]:
    report: dict[str, str] = {}
    for p in ALL_PARAMS:
        val = features.get(p.key)
        if p.key in provided_keys and val is not None:
            report[p.key] = "live"
        elif val is None:
            report[p.key] = "missing"
        elif p.domain == "weather" and p.key not in provided_keys:
            report[p.key] = "weather_default"
        elif p.key == "soil_ph" and p.key not in provided_keys:
            report[p.key] = "lab_only"
        else:
            report[p.key] = p.status
    return report


def split_domains(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    soil_keys = {p.key for p in ALL_PARAMS if p.domain == "soil"}
    weather_keys = {p.key for p in ALL_PARAMS if p.domain == "weather"}
    return {
        "soil": {k: features[k] for k in soil_keys if k in features and features[k] is not None},
        "weather": {
            k: features[k] for k in weather_keys if k in features and features[k] is not None
        },
    }
