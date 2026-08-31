from __future__ import annotations

from app.config import CROP_ALIASES, CROP_DISPLAY_NAMES


def normalize_crop_id(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    key = " ".join(key.split())
    if key in CROP_ALIASES:
        return CROP_ALIASES[key]
    underscored = key.replace(" ", "_")
    return CROP_ALIASES.get(underscored)


def display_crop_name(crop_id: str) -> str:
    return CROP_DISPLAY_NAMES.get(crop_id, crop_id.replace("_", " ").title())
