"""Weather client: OpenWeatherMap when keyed, otherwise Open-Meteo (free)."""
from __future__ import annotations

import os
from typing import Any

import requests

from app.features.vector import estimate_et0, estimate_gdd

OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_api_key() -> str | None:
    return os.environ.get("OPENWEATHERMAP_API_KEY") or os.environ.get("OWM_API_KEY")


def _weather_features(
    *,
    temp: float,
    humidity: float,
    wind: float,
    precip_mm: float,
    temp_max: float,
    temp_min: float,
    horizon_days: float,
) -> dict[str, Any]:
    gdd = estimate_gdd(temp, temp_max, temp_min, days=horizon_days)
    et0 = estimate_et0(temp, humidity, wind)
    return {
        "temperature_c": temp,
        "temp_max_c": temp_max,
        "temp_min_c": temp_min,
        "relative_humidity": humidity,
        "wind_speed": wind,
        "precip_accum_mm": round(precip_mm, 2),
        "gdd": gdd,
        "et0_mm": et0,
    }


def fetch_openweathermap(
    lat: float,
    lon: float,
    *,
    api_key: str,
    timeout: float = 12.0,
) -> dict[str, Any]:
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    cur = requests.get(OWM_WEATHER_URL, params=params, timeout=timeout)
    cur.raise_for_status()
    current = cur.json()

    fc = requests.get(OWM_FORECAST_URL, params=params, timeout=timeout)
    fc.raise_for_status()
    forecast = fc.json()

    temp = float(current["main"]["temp"])
    humidity = float(current["main"]["humidity"])
    wind = float(current.get("wind", {}).get("speed") or 0.0)
    rain_now = 0.0
    if isinstance(current.get("rain"), dict):
        rain_now = float(current["rain"].get("1h") or current["rain"].get("3h") or 0.0)

    temps: list[float] = []
    for item in forecast.get("list", []):
        main = item.get("main") or {}
        temps.append(float(main.get("temp", temp)))
        rain = item.get("rain") or {}
        rain_now += float(rain.get("3h") or 0.0)
        snow = item.get("snow") or {}
        rain_now += float(snow.get("3h") or 0.0)

    temp_max = max(temps) if temps else float(current["main"].get("temp_max", temp))
    temp_min = min(temps) if temps else float(current["main"].get("temp_min", temp))
    horizon_days = max(1.0, len(temps) * 3.0 / 24.0)
    features = _weather_features(
        temp=temp,
        humidity=humidity,
        wind=wind,
        precip_mm=rain_now,
        temp_max=temp_max,
        temp_min=temp_min,
        horizon_days=horizon_days,
    )
    meta = {
        "provider": "openweathermap",
        "coords": {"lat": lat, "lon": lon},
        "current_summary": (current.get("weather") or [{}])[0].get("description"),
        "forecast_steps": len(temps),
        "horizon_days": round(horizon_days, 2),
        "raw_current_keys": list(current.keys()),
    }
    return {"weather": features, "openweathermap": meta, "provider": "openweathermap"}


def fetch_open_meteo(lat: float, lon: float, *, timeout: float = 12.0) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Africa/Kigali",
        "forecast_days": 5,
        "wind_speed_unit": "ms",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current") or {}
    daily = data.get("daily") or {}

    temp = float(current.get("temperature_2m"))
    humidity = float(current.get("relative_humidity_2m") or 70.0)
    wind = float(current.get("wind_speed_10m") or 0.0)
    rain_now = float(current.get("precipitation") or 0.0)
    tmax = [float(x) for x in (daily.get("temperature_2m_max") or [temp])]
    tmin = [float(x) for x in (daily.get("temperature_2m_min") or [temp])]
    precip = [float(x) for x in (daily.get("precipitation_sum") or [rain_now])]
    horizon_days = max(1.0, float(len(tmax)))
    features = _weather_features(
        temp=temp,
        humidity=humidity,
        wind=wind,
        precip_mm=sum(precip),
        temp_max=max(tmax),
        temp_min=min(tmin),
        horizon_days=horizon_days,
    )
    meta = {
        "provider": "open-meteo",
        "coords": {"lat": lat, "lon": lon},
        "current_summary": "open-meteo forecast",
        "forecast_steps": len(tmax),
        "horizon_days": round(horizon_days, 2),
        "timezone": data.get("timezone") or "Africa/Kigali",
    }
    return {"weather": features, "openweathermap": meta, "provider": "open-meteo"}


def fetch_weather_bundle(
    lat: float,
    lon: float,
    *,
    api_key: str | None = None,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """
    Live weather for lat/lon. Uses OpenWeatherMap when a key is set,
    otherwise Open-Meteo so production still gets Rwanda weather.
    """
    key = api_key or get_api_key()
    errors: list[str] = []
    if key:
        try:
            return fetch_openweathermap(lat, lon, api_key=key, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openweathermap: {exc}")
    try:
        return fetch_open_meteo(lat, lon, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"open-meteo: {exc}")
        raise ValueError("Weather fetch failed (" + "; ".join(errors) + ")") from exc
