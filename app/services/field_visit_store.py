"""Store field visits for continuous learning and later retraining."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

FIELD_VISITS_DIR = DATA_DIR / "field_visits"
MANIFEST_PATH = FIELD_VISITS_DIR / "manifest.jsonl"


def save_field_visit(
    *,
    payload: dict[str, Any],
    validated_reading: dict[str, Any],
    analysis_summary: dict[str, Any],
    confidence: dict[str, Any],
) -> Path:
    """Append structured visit record for future model improvement."""
    FIELD_VISITS_DIR.mkdir(parents=True, exist_ok=True)
    device_id = (payload.get("device_id") or "unknown").replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FIELD_VISITS_DIR / f"{stamp}_{device_id}.json"

    record = {
        "visit_id": f"{stamp}_{device_id}",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "device_id": payload.get("device_id"),
        "farm_id": payload.get("farm_id") or payload.get("location_id"),
        "coordinates": payload.get("coordinates"),
        "sensor_reading": validated_reading,
        "ai_summary": analysis_summary,
        "confidence": confidence,
        "outcome_fields_for_later": {
            "crop_planted": None,
            "yield_observed": None,
            "farmer_feedback": None,
            "lab_comparison": None,
        },
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "visit_id": record["visit_id"],
                    "path": str(path.relative_to(DATA_DIR)),
                    "captured_at": record["captured_at"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return path
