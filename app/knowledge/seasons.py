"""Rwanda agricultural seasons (MINAGRI / NISR SAS)."""
from __future__ import annotations

from datetime import date

# Season A: Sep-Feb  Season B: Mar-Jun  Season C: Jul-Aug (marshland / vegetables)
SEASON_MONTHS = {
    "A": (9, 10, 11, 12, 1, 2),
    "B": (3, 4, 5, 6),
    "C": (7, 8),
}

SEASON_LABELS = {
    "A": "Season A (Sep-Feb short rains / main planting)",
    "B": "Season B (Mar-Jun long rains)",
    "C": "Season C (Jul-Aug marshland and vegetables)",
}

SEASON_ORDER = ("A", "B", "C")


def season_from_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return "A"


def current_season(today: date | None = None) -> str:
    return season_from_month((today or date.today()).month)


def next_seasons(start: str, count: int = 3) -> list[str]:
    idx = SEASON_ORDER.index(start) if start in SEASON_ORDER else 0
    return [SEASON_ORDER[(idx + i + 1) % 3] for i in range(count)]


def infer_province(lat: float | None, lon: float | None) -> str | None:
    """Coarse province from GPS. Rwanda ~ lat -2.85..-1.05, lon 28.85..30.90."""
    if lat is None or lon is None:
        return None
    if not (-3.2 <= lat <= -0.8 and 28.6 <= lon <= 31.2):
        return None
    if lon < 29.35:
        return "Western"
    if lat > -1.72 and lon < 30.05:
        return "Northern"
    if lon >= 30.18:
        return "Eastern"
    if lat > -2.05 and 29.9 <= lon < 30.18:
        return "Kigali"
    return "Southern"
