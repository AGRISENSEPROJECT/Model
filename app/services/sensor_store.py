"""Persist device sensor payloads and AI results for audit/review."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

READINGS_DIR = DATA_DIR / "device_readings"


def save_device_reading(payload: dict[str, Any], analysis: dict[str, Any]) -> Path:
    READINGS_DIR.mkdir(parents=True, exist_ok=True)
    device_id = (payload.get("device_id") or "unknown").replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = READINGS_DIR / f"{stamp}_{device_id}.json"
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "analysis": analysis,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
