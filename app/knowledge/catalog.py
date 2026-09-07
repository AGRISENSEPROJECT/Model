"""Single crop catalog: agronomy, prices, and Rwanda suitability.

Source of truth: data/rwanda/crop_recommendation_factors.csv
(NISR SAS yields, FAOSTAT/MINAGRI farmgate prices, RAB season notes).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

FACTORS_CSV = DATA_DIR / "rwanda" / "crop_recommendation_factors.csv"


def _split(value: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in (value or "").split("|") if p.strip())


def _bool(value: str) -> bool:
    return str(value).strip() in {"1", "true", "yes"}


def _row_to_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "crop_id": row["crop_id"].strip(),
        "display_name": row["display_name"].strip(),
        "family": row["family"].strip(),
        "seasons": _split(row.get("seasons", "")),
        "soils": _split(row.get("soils", "")),
        "temp_min": float(row["temp_min"]),
        "temp_max": float(row["temp_max"]),
        "rain_min": float(row["rain_min"]),
        "rain_max": float(row["rain_max"]),
        "humidity_min": float(row["humidity_min"]),
        "humidity_max": float(row["humidity_max"]),
        "ph_min": float(row["ph_min"]),
        "ph_max": float(row["ph_max"]),
        "n_opt": float(row["n_opt"]),
        "p_opt": float(row["p_opt"]),
        "k_opt": float(row["k_opt"]),
        "typical_yield_t_ha": float(row["typical_yield_t_ha"]),
        "farmgate_rwf_kg": float(row["farmgate_rwf_kg"]),
        "price_source": row.get("price_source", ""),
        "rwanda_priority": float(row.get("rwanda_priority") or 1),
        "ml_backed": _bool(row.get("ml_backed", "0")),
        "perennial": _bool(row.get("perennial", "0")),
        "provinces": _split(row.get("provinces", "")),
        "notes": row.get("notes", ""),
    }


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, Any]]:
    if not FACTORS_CSV.exists():
        raise FileNotFoundError(f"Missing crop factor table: {FACTORS_CSV}")
    out: dict[str, dict[str, Any]] = {}
    with FACTORS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rec = _row_to_record(row)
            out[rec["crop_id"]] = rec
    return out


def all_crop_ids() -> tuple[str, ...]:
    return tuple(load_catalog().keys())


def suitability_map() -> dict[str, dict[str, Any]]:
    """Shape expected by the legacy rule fallback."""
    mapped = {}
    for crop_id, rec in load_catalog().items():
        mapped[crop_id] = {
            "soil_texture": list(rec["soils"]),
            "temperature": {"min": rec["temp_min"], "max": rec["temp_max"]},
            "humidity": {"min": rec["humidity_min"], "max": rec["humidity_max"]},
            "rainfall": {"min": rec["rain_min"], "max": rec["rain_max"]},
        }
    return mapped
