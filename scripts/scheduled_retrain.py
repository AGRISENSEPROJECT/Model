#!/usr/bin/env python3
"""
Scheduled soil-model retrain gate.

Runs safely from cron every day; only retrains when:
  - last retrain was ≥ RETRAIN_INTERVAL_SECONDS ago (default 90 days), OR
  - human_labeled count ≥ RETRAIN_MIN_NEW_LABELED (default 50)

Then: promote gold human labels → data/train → train_soil_texture → evaluate.
No auto/pseudo labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrain_config import (
    RETRAIN_INTERVAL_SECONDS,
    RETRAIN_MIN_NEW_LABELED,
    RETRAIN_STATE,
)
from app.services.feedback_store import ensure_retrain_dirs, retrain_status
from scripts.promote_feedback_to_dataset import promote


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def should_retrain(force: bool = False) -> tuple[bool, str]:
    ensure_retrain_dirs()
    status = retrain_status()
    state = status["state"]
    labeled = status["human_labeled_total"]

    if force:
        return True, "forced"

    last = _parse_ts(state.get("last_retrain_at"))
    now = datetime.now(timezone.utc)
    if last is None:
        if labeled >= RETRAIN_MIN_NEW_LABELED:
            return True, f"first_run_with_{labeled}_human_labels"
        return False, f"waiting_for_{RETRAIN_MIN_NEW_LABELED}_human_labels (have {labeled})"

    elapsed = (now - last).total_seconds()
    if elapsed >= RETRAIN_INTERVAL_SECONDS and labeled > 0:
        return True, f"interval_elapsed_{int(elapsed)}s_with_{labeled}_labels"
    if labeled >= RETRAIN_MIN_NEW_LABELED:
        return True, f"label_threshold_{labeled}"
    days_left = max(0, int((RETRAIN_INTERVAL_SECONDS - elapsed) / 86400))
    return False, f"not_due (days_left≈{days_left}, human_labeled={labeled})"


def run(force: bool = False, epochs: int = 15) -> dict:
    ok, reason = should_retrain(force=force)
    report = {
        "should_retrain": ok,
        "reason": reason,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if not ok:
        print(json.dumps(report, indent=2))
        return report

    promo = promote(dest_split="train")
    report["promote"] = promo

    from scripts.evaluate_soil_model import execute_validation_audit
    from scripts.train_soil_texture import train as train_soil

    train_soil(epochs=epochs)
    metrics = execute_validation_audit()
    from app.services.production_ready import soil_retrain_gate

    gate = soil_retrain_gate(metrics)
    report["validation_accuracy"] = metrics.get("accuracy")
    report["sandy_recall"] = gate["sandy_recall"]
    report["promotion_gate"] = gate
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    state = {
        "last_retrain_at": report["finished_at"],
        "last_retrain_status": "ok" if gate["ok"] else "rejected_gate",
        "last_reason": reason,
        "last_validation_accuracy": metrics.get("accuracy"),
        "last_sandy_recall": gate["sandy_recall"],
        "last_promote": promo,
        "gate_reasons": gate["reasons"],
    }
    RETRAIN_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if not gate["ok"]:
        report["warning"] = (
            "Retrain finished but was not promoted as production-ready: "
            + "; ".join(gate["reasons"])
        )
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        ok, reason = should_retrain(force=args.force)
        print(
            json.dumps(
                {"should_retrain": ok, "reason": reason, **retrain_status()},
                indent=2,
                default=str,
            )
        )
        return
    run(force=args.force, epochs=args.epochs)


if __name__ == "__main__":
    main()
