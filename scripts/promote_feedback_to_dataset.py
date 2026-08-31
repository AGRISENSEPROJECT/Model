#!/usr/bin/env python3
"""
Promote human-labeled images into data/train.

Does NOT retrain by itself — call scheduled_retrain.py or train_soil_texture.py after.
Manual labels only (no auto/pseudo labels).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.retrain_config import RETRAIN_LABELED, SOIL_CLASSES
from app.services.feedback_store import ensure_retrain_dirs


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def promote(dest_split: str = "train") -> dict:
    ensure_retrain_dirs()
    dest_root = DATA_DIR / dest_split
    copied = {cls: 0 for cls in SOIL_CLASSES}

    for cls in SOIL_CLASSES:
        src_dir = RETRAIN_LABELED / cls
        out_dir = dest_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            continue
        for img in src_dir.iterdir():
            if not img.is_file() or img.suffix.lower() in {".json", ".txt"}:
                continue
            name = f"retrain_human_{_file_hash(img)}_{img.name}"
            target = out_dir / name
            if target.exists():
                continue
            shutil.copy2(img, target)
            copied[cls] += 1

    return {
        "dest_split": dest_split,
        "copied_by_class": copied,
        "copied_total": sum(copied.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()
    print(promote(dest_split=args.split))


if __name__ == "__main__":
    main()
