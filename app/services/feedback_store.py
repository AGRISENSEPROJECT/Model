"""
Capture prediction images for future scheduled retraining.

Flow
----
1. Every soil image used in /predict → data/retrain/inbox/ + manifest.jsonl
2. Human correction via API → human_labeled/{class}/  (gold labels only)
3. scripts/scheduled_retrain.py merges gold labels into data/train and retrains
   when 90 days elapsed OR enough new human labels exist.

No automatic pseudo-labeling — labels are manual only.
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR
from app.retrain_config import (
    RETRAIN_INBOX,
    RETRAIN_LABELED,
    RETRAIN_MANIFEST,
    RETRAIN_REJECTED,
    RETRAIN_ROOT,
    RETRAIN_STATE,
    SOIL_CLASSES,
)


def ensure_retrain_dirs() -> None:
    for base in (RETRAIN_INBOX, RETRAIN_LABELED, RETRAIN_REJECTED):
        base.mkdir(parents=True, exist_ok=True)
    for cls in SOIL_CLASSES:
        (RETRAIN_LABELED / cls).mkdir(parents=True, exist_ok=True)
    RETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    if not RETRAIN_STATE.exists():
        RETRAIN_STATE.write_text(
            json.dumps(
                {
                    "last_retrain_at": None,
                    "last_retrain_status": "never",
                    "total_inbox": 0,
                    "total_human_labeled": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _append_manifest(record: dict[str, Any]) -> None:
    ensure_retrain_dirs()
    with RETRAIN_MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def archive_prediction_image(
    image_path: str | Path,
    *,
    predicted_texture: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
    request_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a production prediction photo for later manual labeling + retrain."""
    ensure_retrain_dirs()
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Cannot archive missing image: {src}")

    pred = (predicted_texture or "unknown").strip().lower()
    ts = datetime.now(timezone.utc)
    sample_id = f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"
    suffix = src.suffix.lower() or ".jpg"
    inbox_name = f"{sample_id}{suffix}"
    inbox_path = RETRAIN_INBOX / inbox_name
    shutil.copy2(src, inbox_path)

    try:
        inbox_rel = str(inbox_path.relative_to(ROOT_DIR))
    except ValueError:
        inbox_rel = str(inbox_path)

    sidecar = {
        "sample_id": sample_id,
        "captured_at": ts.isoformat(),
        "inbox_path": inbox_rel,
        "source_path": str(src),
        "predicted_texture": pred,
        "confidence": float(confidence),
        "probabilities": probabilities or {},
        "label_status": "unverified",
        "human_label": None,
        "request_meta": request_meta or {},
    }
    (RETRAIN_INBOX / f"{sample_id}.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    _append_manifest(sidecar)
    return sidecar


def apply_human_label(sample_id: str, human_label: str, notes: str | None = None) -> dict[str, Any]:
    """Promote an inbox sample to gold human_labeled/{class}/."""
    ensure_retrain_dirs()
    label = human_label.strip().lower()
    if label not in SOIL_CLASSES:
        raise ValueError(f"human_label must be one of {SOIL_CLASSES}")

    sidecar_path = RETRAIN_INBOX / f"{sample_id}.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Unknown sample_id: {sample_id}")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    matches = list(RETRAIN_INBOX.glob(f"{sample_id}.*"))
    image = next((p for p in matches if p.suffix.lower() != ".json"), None)
    if image is None:
        raise FileNotFoundError(f"Image for sample_id {sample_id} missing in inbox")

    dest = RETRAIN_LABELED / label / image.name
    shutil.copy2(image, dest)
    sidecar.update(
        {
            "human_label": label,
            "label_status": "human_verified",
            "human_labeled_path": str(dest),
            "human_notes": notes,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    _append_manifest({"event": "human_label", **sidecar})
    return sidecar


def reject_sample(sample_id: str, reason: str = "rejected") -> dict[str, Any]:
    ensure_retrain_dirs()
    sidecar_path = RETRAIN_INBOX / f"{sample_id}.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Unknown sample_id: {sample_id}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    matches = list(RETRAIN_INBOX.glob(f"{sample_id}.*"))
    image = next((p for p in matches if p.suffix.lower() != ".json"), None)
    if image:
        dest = RETRAIN_REJECTED / image.name
        shutil.move(str(image), str(dest))
        sidecar["rejected_path"] = str(dest)
    sidecar["label_status"] = "rejected"
    sidecar["reject_reason"] = reason
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    _append_manifest({"event": "reject", **sidecar})
    return sidecar


def retrain_status() -> dict[str, Any]:
    ensure_retrain_dirs()
    state = json.loads(RETRAIN_STATE.read_text(encoding="utf-8"))
    labeled_counts = {
        cls: len(list((RETRAIN_LABELED / cls).glob("*")))
        for cls in SOIL_CLASSES
    }
    inbox_images = [
        p for p in RETRAIN_INBOX.iterdir() if p.suffix.lower() != ".json" and p.is_file()
    ]
    return {
        "state": state,
        "inbox_images": len(inbox_images),
        "human_labeled_by_class": labeled_counts,
        "human_labeled_total": sum(labeled_counts.values()),
        "auto_labeling": False,
        "paths": {
            "inbox": str(RETRAIN_INBOX),
            "human_labeled": str(RETRAIN_LABELED),
            "manifest": str(RETRAIN_MANIFEST),
        },
    }
