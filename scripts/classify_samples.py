#!/usr/bin/env python3
"""Classify images in data/samples/pending_review into texture folders using the soil model."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.services.soil_texture import predict_texture


def classify_pending(move: bool = False, min_confidence: float = 0.55) -> list[dict]:
    pending = DATA_DIR / "samples" / "pending_review"
    results = []
    if not pending.exists():
        print(f"No pending folder at {pending}")
        return results

    for path in sorted(pending.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            pred = predict_texture(path)
        except Exception as exc:  # noqa: BLE001
            results.append({"file": path.name, "error": str(exc)})
            continue

        texture = pred["texture"]
        conf = pred["confidence"]
        entry = {
            "file": path.name,
            "texture": texture,
            "confidence": conf,
            "probabilities": pred["probabilities"],
            "action": "none",
        }
        dest_dir = DATA_DIR / "samples" / texture
        dest_dir.mkdir(parents=True, exist_ok=True)
        if move and conf >= min_confidence:
            dest = dest_dir / path.name
            shutil.move(str(path), str(dest))
            entry["action"] = f"moved → {dest}"
        elif conf < min_confidence:
            entry["action"] = "left in pending (low confidence)"
        else:
            entry["action"] = f"would move → {dest_dir / path.name} (dry-run)"
        results.append(entry)
        print(entry)

    manifest = DATA_DIR / "samples" / "SAMPLE_MANIFEST.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {manifest}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move", action="store_true", help="Move files into class folders")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()
    classify_pending(move=args.move, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
