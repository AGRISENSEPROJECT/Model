"""OpenWeatherMap current + 5-day/3-hour forecast client."""
from __future__ import annotations

import os
from typing import Any

import requests

from app.features.vector import estimate_et0, estimate_gdd

OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_api_key() -> str | None:
    return os.environ.get("OPENWEATHERMAP_API_KEY") or os.environ.get("OWM_API_KEY")


def fetch_weather_bundle(
    lat: float,
    lon: float,
    *,
    api_key: str | None = None,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """
    Pull current weather + forecast and map to lean weather features.
    Raises ValueError if API key missing or request fails.
    """
    key = api_key or get_api_key()
    if not key:
        raise ValueError(
            "Missing OPENWEATHERMAP_API_KEY (or OWM_API_KEY) environment variable"
        )

    params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
    cur = requests.get(OWM_WEATHER_URL, params=params, timeout=timeout)
    cur.raise_for_status()
    current = cur.json()

    fc = requests.get(OWM_FORECAST_URL, params=params, timeout=timeout)
    fc.raise_for_status()
    forecast = fc.json()

    temp = float(current["main"]["temp"])
    humidity = float(current["main"]["humidity"])
    wind = float(current.get("wind", {}).get("speed") or 0.0)
    # current rain may be 1h or 3h
    rain_now = 0.0
    if isinstance(current.get("rain"), dict):
        rain_now = float(current["rain"].get("1h") or current["rain"].get("3h") or 0.0)

    temps: list[float] = []
    humidities: list[float] = []
    winds: list[float] = []
    precip_mm = rain_now
    for item in forecast.get("list", []):
        main = item.get("main") or {}
        temps.append(float(main.get("temp", temp)))
        humidities.append(float(main.get("humidity", humidity)))
        winds.append(float((item.get("wind") or {}).get("speed") or wind))
        rain = item.get("rain") or {}
        precip_mm += float(rain.get("3h") or 0.0)
        snow = item.get("snow") or {}
        precip_mm += float(snow.get("3h") or 0.0)

    temp_max = max(temps) if temps else float(current["main"].get("temp_max", temp))
    temp_min = min(temps) if temps else float(current["main"].get("temp_min", temp))
    # ~5 days of 3h steps
    horizon_days = max(1.0, len(temps) * 3.0 / 24.0)
    gdd = estimate_gdd(temp, temp_max, temp_min, days=horizon_days)
    et0 = estimate_et0(temp, humidity, wind)

    weather_features = {
        "temperature_c": temp,
        "temp_max_c": temp_max,
        "temp_min_c": temp_min,
        "relative_humidity": humidity,
        "wind_speed": wind,
        "precip_accum_mm": round(precip_mm, 2),
        "gdd": gdd,
        "et0_mm": et0,
    }

    return {
        "weather": weather_features,
        "openweathermap": {
            "provider": "openweathermap",
            "coords": {"lat": lat, "lon": lon},
            "current_summary": (current.get("weather") or [{}])[0].get("description"),
            "forecast_steps": len(temps),
            "horizon_days": round(horizon_days, 2),
            "raw_current_keys": list(current.keys()),
        },
    }
