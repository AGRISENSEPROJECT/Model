"""Continuous improvement / scheduled retraining paths."""
from __future__ import annotations

import os
from pathlib import Path

from app.config import DATA_DIR, ROOT_DIR

# Production photos captured during /predict (raw + sidecars)
RETRAIN_ROOT = Path(os.environ.get("AGRISENSE_RETRAIN_ROOT", DATA_DIR / "retrain"))
RETRAIN_INBOX = RETRAIN_ROOT / "inbox"  # every prediction image lands here
RETRAIN_PSEUDO = RETRAIN_ROOT / "pseudo_labeled"  # high-confidence model labels (weak)
RETRAIN_LABELED = RETRAIN_ROOT / "human_labeled"  # agronomist-corrected (gold)
RETRAIN_REJECTED = RETRAIN_ROOT / "rejected"
RETRAIN_MANIFEST = RETRAIN_ROOT / "manifest.jsonl"
RETRAIN_STATE = RETRAIN_ROOT / "retrain_state.json"

# How often automatic retrain is allowed (seconds) — default 90 days
RETRAIN_INTERVAL_SECONDS = int(
    os.environ.get("AGRISENSE_RETRAIN_INTERVAL_SECONDS", str(90 * 24 * 3600))
)
# Minimum new human-labeled images before a scheduled retrain fires early
RETRAIN_MIN_NEW_LABELED = int(os.environ.get("AGRISENSE_RETRAIN_MIN_NEW_LABELED", "50"))
# Pseudo-label confidence floor (do not treat lower as training data)
PSEUDO_LABEL_MIN_CONFIDENCE = float(
    os.environ.get("AGRISENSE_PSEUDO_MIN_CONFIDENCE", "0.85")
)

SOIL_CLASSES = ("alluvial", "clayey", "loamy", "sandy")
