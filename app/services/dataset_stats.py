"""Training-dataset statistics for data-driven nutrient and soil assessments."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import DATA_DIR, ENV_DATASET_PATH

STATS_CACHE_PATH = DATA_DIR / "environmental" / "nutrient_percentiles.json"

_lock = threading.Lock()
_cache: dict[str, Any] | None = None

METRIC_COLUMNS = {
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
    "ec_us_cm": None,  # not in CSV — built from NPK spread proxy until field EC labels exist
}


def _ec_proxy(row: pd.Series) -> float:
    """Placeholder spread until real EC labels exist — not used for absolute EC advice."""
    return float((row["N"] + row["P"] + row["K"]) / 3.0)


def _build_stats() -> dict[str, Any]:
    if not ENV_DATASET_PATH.exists():
        return {"source": str(ENV_DATASET_PATH), "available": False, "global": {}, "by_crop": {}}

    df = pd.read_csv(ENV_DATASET_PATH)
    df = df.dropna(subset=["N", "P", "K"])

    def percentiles(series: pd.Series) -> dict[str, float]:
        return {
            "p10": float(series.quantile(0.10)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    global_stats: dict[str, Any] = {}
    for key, col in METRIC_COLUMNS.items():
        if col and col in df.columns:
            global_stats[key] = percentiles(df[col])

    by_crop: dict[str, Any] = {}
    for crop, group in df.groupby("Crop_Type"):
        by_crop[str(crop)] = {
            "nitrogen": percentiles(group["N"]),
            "phosphorus": percentiles(group["P"]),
            "potassium": percentiles(group["K"]),
        }

    return {
        "source": str(ENV_DATASET_PATH),
        "available": True,
        "n_rows": int(len(df)),
        "global": global_stats,
        "by_crop": by_crop,
        "note": "Percentiles from Environmental Factors CSV; EC requires local calibration dataset",
    }


def get_dataset_stats() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is None:
            if STATS_CACHE_PATH.exists():
                try:
                    _cache = json.loads(STATS_CACHE_PATH.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    _cache = _build_stats()
            else:
                _cache = _build_stats()
    return _cache


def classify_against_distribution(
    value: float,
    stats: dict[str, float],
) -> tuple[str, float]:
    """
    Classify a reading vs training distribution.
    Returns (label, percentile_estimate).
    """
    p10, p25, p50, p75, p90 = (
        stats["p10"],
        stats["p25"],
        stats["p50"],
        stats["p75"],
        stats["p90"],
    )
    if value <= p10:
        return "very_low", 5.0
    if value <= p25:
        return "low", 17.5
    if value <= p50:
        return "moderate", 37.5
    if value <= p75:
        return "adequate", 62.5
    if value <= p90:
        return "high", 82.5
    return "very_high", 95.0
