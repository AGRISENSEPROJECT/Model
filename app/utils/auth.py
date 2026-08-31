"""Lightweight API-key checks for device-facing routes."""
from __future__ import annotations

import os

from flask import request


def sensor_api_key_valid() -> bool:
    expected = os.environ.get("SENSOR_API_KEY", "").strip()
    if not expected:
        return True
    provided = (request.headers.get("X-API-Key") or "").strip()
    return provided == expected


def sensor_api_key_error() -> str:
    return "Missing or invalid X-API-Key header"
