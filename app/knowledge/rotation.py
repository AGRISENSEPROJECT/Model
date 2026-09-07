"""Crop-family rotation rules for Rwanda mixed farms."""
from __future__ import annotations

from typing import Any

SOLANACEAE = {"irish_potatoes", "tomatoes"}

FAMILY_FOLLOW = {
    ("cereal", "legume"): (
        1.15,
        "Legume after a cereal restores nitrogen and breaks pest cycles",
    ),
    ("legume", "cereal"): (
        1.1,
        "Cereal after a legume uses residual nitrogen",
    ),
    ("legume", "tuber"): (
        1.05,
        "Tuber after a legume is a sound fertility sequence",
    ),
    ("tuber", "legume"): (
        1.08,
        "Legume after a tuber rests the soil and adds nitrogen",
    ),
    ("cereal", "vegetable"): (
        0.95,
        "Acceptable, watch fertility and irrigation",
    ),
    ("vegetable", "legume"): (
        1.08,
        "Legume after vegetables rebuilds soil nitrogen",
    ),
}


def rotation_fit(previous_id: str | None, candidate_id: str, catalog: dict[str, Any]) -> tuple[float, str]:
    if not previous_id:
        return 0.78, "No previous crop given — rotation not constrained"
    prev = catalog.get(previous_id)
    cand = catalog.get(candidate_id)
    if not prev or not cand:
        return 0.75, "Unknown previous crop"
    if cand.get("perennial"):
        return 1.0, "Perennial crop — rotation does not apply once established"
    if previous_id == candidate_id:
        return 0.28, "Avoid planting the same crop back-to-back (pests and nutrient mining)"
    if previous_id in SOLANACEAE and candidate_id in SOLANACEAE:
        return 0.32, "Do not follow potato with tomato (or the reverse) — shared blight risk"
    if prev.get("family") == cand.get("family") and not cand.get("perennial"):
        return 0.42, f"Same family ({cand['family']}) — higher disease and nutrient risk"
    key = (prev.get("family"), cand.get("family"))
    if key in FAMILY_FOLLOW:
        score, reason = FAMILY_FOLLOW[key]
        return score, reason
    return 0.88, "Different families — acceptable rotation"
