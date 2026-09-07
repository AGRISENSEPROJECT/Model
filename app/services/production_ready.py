"""Production readiness checks. No TensorFlow import."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    ENV_META_PATH,
    SOIL_CLASS_INDICES_PATH,
    SOIL_MODEL_PATH,
    YIELD_MODEL_PATH,
)
from app.services.confidence import CNN_CONFIDENCE_FLOOR
from app.services.precision_recommender import PRECISION_META_PATH, PRECISION_MODEL_PATH
from app.services.weather import get_api_key

CLASSES = ("alluvial", "clayey", "loamy", "sandy")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MIN_SANDY_TRAIN = 200
MIN_SANDY_VAL = 24
MIN_SANDY_RECALL = 0.50
MIN_SOIL_ACCURACY = 0.80


def _count_split(split: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for klass in CLASSES:
        folder = split / klass
        counts[klass] = (
            len([p for p in folder.iterdir() if p.suffix.lower() in EXTS])
            if folder.exists()
            else 0
        )
    return counts


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def production_status() -> dict[str, Any]:
    train_counts = _count_split(DATA_DIR / "train")
    val_counts = _count_split(DATA_DIR / "validation")
    report = _load_json(ARTIFACTS_DIR / "soil_validation_report.json") or {}
    class_report = report.get("report") or {}
    sandy = class_report.get("sandy") or {}
    sandy_recall = float(sandy.get("recall") or 0.0)
    soil_accuracy = float(report.get("accuracy") or 0.0)

    artifacts = {
        "soil_cnn": SOIL_MODEL_PATH.exists(),
        "soil_class_indices": SOIL_CLASS_INDICES_PATH.exists(),
        "soil_validation_report": (ARTIFACTS_DIR / "soil_validation_report.json").exists(),
        "yield_predictor": YIELD_MODEL_PATH.exists(),
        "environmental_meta": ENV_META_PATH.exists(),
        "precision_ranker": PRECISION_MODEL_PATH.exists(),
        "precision_meta": PRECISION_META_PATH.exists(),
    }
    missing_artifacts = [name for name, ok in artifacts.items() if not ok]

    warnings: list[str] = []
    if train_counts.get("sandy", 0) < MIN_SANDY_TRAIN:
        warnings.append(
            f"Sandy train images {train_counts.get('sandy', 0)} < {MIN_SANDY_TRAIN}"
        )
    if val_counts.get("sandy", 0) < MIN_SANDY_VAL:
        warnings.append(
            f"Sandy val images {val_counts.get('sandy', 0)} < {MIN_SANDY_VAL}"
        )
    if sandy_recall < MIN_SANDY_RECALL:
        warnings.append(
            f"Sandy recall {sandy_recall:.1%} is below production floor {MIN_SANDY_RECALL:.0%}"
        )
    if soil_accuracy and soil_accuracy < MIN_SOIL_ACCURACY:
        warnings.append(f"Soil CNN accuracy {soil_accuracy:.1%} is below {MIN_SOIL_ACCURACY:.0%}")
    if missing_artifacts:
        warnings.append("Missing artifacts: " + ", ".join(missing_artifacts))

    weather_owm = bool(get_api_key())
    ready = (
        not missing_artifacts
        and train_counts.get("sandy", 0) >= MIN_SANDY_TRAIN
        and sandy_recall >= MIN_SANDY_RECALL
        and (not soil_accuracy or soil_accuracy >= MIN_SOIL_ACCURACY)
    )

    return {
        "status": "ok" if ready else "degraded",
        "production_ready": ready,
        "service": "agrisense",
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "dataset": {"train": train_counts, "validation": val_counts},
        "soil_cnn": {
            "accuracy": round(soil_accuracy, 4) if soil_accuracy else None,
            "sandy_recall": round(sandy_recall, 4),
            "sandy_f1": round(float(sandy.get("f1-score") or 0.0), 4),
            "sandy_support": sandy.get("support"),
            "min_sandy_recall": MIN_SANDY_RECALL,
            "cnn_confidence_floor": CNN_CONFIDENCE_FLOOR,
        },
        "weather": {
            "openweathermap_configured": weather_owm,
            "open_meteo_fallback": True,
        },
        "warnings": warnings,
    }


def soil_retrain_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Reject promotion when sandy recall or overall accuracy is too low."""
    sandy = (metrics.get("report") or {}).get("sandy") or {}
    recall = float(sandy.get("recall") or 0.0)
    accuracy = float(metrics.get("accuracy") or 0.0)
    ok = accuracy >= MIN_SOIL_ACCURACY and recall >= MIN_SANDY_RECALL
    reasons: list[str] = []
    if accuracy < MIN_SOIL_ACCURACY:
        reasons.append(f"accuracy {accuracy:.3f} < {MIN_SOIL_ACCURACY}")
    if recall < MIN_SANDY_RECALL:
        reasons.append(f"sandy_recall {recall:.3f} < {MIN_SANDY_RECALL}")
    return {
        "ok": ok,
        "accuracy": accuracy,
        "sandy_recall": recall,
        "reasons": reasons,
    }
