from __future__ import annotations

from typing import Any

from app.knowledge.diseases import CROP_DISEASES, DISEASE_PATTERNS
from app.utils.crops import normalize_crop_id


def detect_disease(image_path: str, crop_type: str) -> dict[str, Any]:
    """Placeholder until a plant-disease CNN / satellite NDVI path is trained."""
    crop_id = normalize_crop_id(crop_type) or "unknown"
    known = CROP_DISEASES.get(crop_id, [])
    patterns = {name: DISEASE_PATTERNS[name] for name in known if name in DISEASE_PATTERNS}
    return {
        "status": "model_pending",
        "message": "Disease CNN not trained yet — returning knowledge catalog only",
        "crop_id": crop_id,
        "image_received": bool(image_path),
        "available_diseases": known,
        "known_patterns": patterns,
        "health_status": "unknown",
        "detected_diseases": {},
    }
