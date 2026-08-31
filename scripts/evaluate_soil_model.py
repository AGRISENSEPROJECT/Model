#!/usr/bin/env python3
"""Validation audit for soil texture model — accuracy, precision, recall, F1, confusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR, IMG_SIZE, SOIL_CLASS_INDICES_PATH, SOIL_MODEL_PATH
from app.services.soil_texture import load_class_indices, save_class_indices


def execute_validation_audit(
    model_path: Path = SOIL_MODEL_PATH,
    val_dir: Path = DATA_DIR / "validation",
) -> dict:
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation dir not found: {val_dir}")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    print("\n--- SYSTEM CORE CONFIGURATIONS ---")
    print("Keras Detected Directory Order mapping:", val_set.class_indices)
    save_class_indices(val_set.class_indices, SOIL_CLASS_INDICES_PATH)
    print(f"Synced class map → {SOIL_CLASS_INDICES_PATH}")

    runtime_map = load_class_indices()
    print("Runtime inference map:", runtime_map)
    if runtime_map != val_set.class_indices:
        print("WARNING: runtime map differs from validation generator indices!")

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(val_set, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_set.classes
    target_names = list(val_set.class_indices.keys())

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    print("\n--- STATISTICAL CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    )
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred))

    # Simulate OLD bug: wrong hardcoded order
    wrong_order = ["sandy", "loamy", "clayey", "alluvial"]
    wrong_labels = [wrong_order[i] if i < len(wrong_order) else "?" for i in y_pred]
    true_labels = [target_names[i] for i in y_true]
    wrong_acc = sum(a == b for a, b in zip(true_labels, wrong_labels)) / len(true_labels)
    correct_acc = report["accuracy"]
    print("\n--- LABEL-ORDER IMPACT ---")
    print(f"Accuracy with CORRECT alphabetical map: {correct_acc:.4f}")
    print(f"Accuracy if using OLD hardcoded map {wrong_order}: {wrong_acc:.4f}")

    out = {
        "class_indices": val_set.class_indices,
        "accuracy": correct_acc,
        "accuracy_with_old_hardcoded_order": wrong_acc,
        "report": report,
        "confusion_matrix": matrix,
        "target_names": target_names,
    }
    report_path = ROOT / "artifacts" / "soil_validation_report.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SOIL_MODEL_PATH)
    parser.add_argument("--val-dir", type=Path, default=DATA_DIR / "validation")
    args = parser.parse_args()
    execute_validation_audit(model_path=args.model, val_dir=args.val_dir)


if __name__ == "__main__":
    main()
