#!/usr/bin/env python3
"""Train MobileNetV2 soil texture classifier with class weights + saved index map."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    SOIL_CLASS_INDICES_PATH,
    SOIL_MODEL_PATH,
)
from app.services.soil_texture import save_class_indices


class ProductionCheckpoint:
    """Save the epoch that best balances overall accuracy and sandy recall."""

    def __init__(self, val_gen, class_indices: dict[str, int], path: Path):
        import tensorflow as tf

        self.Callback = tf.keras.callbacks.Callback
        self.val_gen = val_gen
        self.sandy_idx = int(class_indices["sandy"])
        self.path = path
        self.best_score = -1.0
        self.best_metrics: dict = {}

    def make(self):
        outer = self

        class _Callback(self.Callback):
            def on_epoch_end(self, epoch, logs=None):
                outer.val_gen.reset()
                preds = self.model.predict(
                    outer.val_gen, steps=len(outer.val_gen), verbose=0
                )
                y_pred = np.argmax(preds, axis=1)
                y_true = outer.val_gen.classes
                acc = float((y_pred == y_true).mean())
                sandy_mask = y_true == outer.sandy_idx
                sandy_n = int(sandy_mask.sum())
                sandy_recall = (
                    float(((y_pred == outer.sandy_idx) & sandy_mask).sum() / sandy_n)
                    if sandy_n
                    else 0.0
                )
                # Production bottleneck is sandy recall; keep overall accuracy in the score.
                score = 0.4 * acc + 0.6 * sandy_recall
                print(
                    f"  production score={score:.4f}  val_acc={acc:.4f}  "
                    f"sandy_recall={sandy_recall:.4f} ({sandy_n} val images)"
                )
                if score > outer.best_score:
                    outer.best_score = score
                    outer.best_metrics = {
                        "epoch": int(epoch) + 1,
                        "score": score,
                        "accuracy": acc,
                        "sandy_recall": sandy_recall,
                    }
                    self.model.save(outer.path)
                    print(f"  saved improved production checkpoint -> {outer.path}")

        return _Callback()


def _backup_existing_model() -> Path | None:
    if not SOIL_MODEL_PATH.exists():
        return None
    backup = SOIL_MODEL_PATH.with_name(SOIL_MODEL_PATH.stem + ".prev.keras")
    shutil.copy2(SOIL_MODEL_PATH, backup)
    print(f"Backed up previous soil model -> {backup}")
    return backup


def train(epochs: int = 24, warmup_epochs: int = 6, sandy_boost: float = 1.8) -> dict:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "validation"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training directory: {train_dir}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _backup_existing_model()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
        channel_shift_range=18.0,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print("Keras class_indices (alphabetical):", train_gen.class_indices)
    save_class_indices(train_gen.class_indices, SOIL_CLASS_INDICES_PATH)
    print(f"Wrote {SOIL_CLASS_INDICES_PATH}")

    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weight_dict = {int(k): float(v) for k, v in zip(np.unique(labels), weights)}
    sandy_idx = int(train_gen.class_indices["sandy"])
    class_weight_dict[sandy_idx] = class_weight_dict[sandy_idx] * sandy_boost
    print("Class weights (sandy boosted):", class_weight_dict)

    n_classes = len(train_gen.class_indices)
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    prod_ckpt = ProductionCheckpoint(val_gen, train_gen.class_indices, SOIL_MODEL_PATH)

    history_all: dict[str, list] = {}

    def _merge_history(hist):
        for key, values in hist.history.items():
            history_all.setdefault(key, []).extend([float(v) for v in values])

    # Phase 1: train the head only so new sandy photos can move the decision surface.
    base.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
    print(f"\n--- Phase 1: frozen backbone, {warmup_epochs} epochs ---")
    hist1 = model.fit(
        train_gen,
        epochs=warmup_epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[prod_ckpt.make()],
    )
    _merge_history(hist1)

    # Phase 2: fine-tune the last convolutional blocks at a low learning rate.
    base.trainable = True
    freeze_until = max(0, len(base.layers) - 40)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= freeze_until
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss, metrics=["accuracy"])
    remaining = max(1, epochs - warmup_epochs)
    print(f"\n--- Phase 2: fine-tune last 40 layers, {remaining} epochs ---")
    hist2 = model.fit(
        train_gen,
        epochs=remaining,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[
            prod_ckpt.make(),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ],
    )
    _merge_history(hist2)

    if SOIL_MODEL_PATH.exists() and prod_ckpt.best_score >= 0:
        print(f"Keeping production checkpoint (score={prod_ckpt.best_score:.4f})")
    else:
        model.save(SOIL_MODEL_PATH)
        print(f"Saved soil model -> {SOIL_MODEL_PATH}")

    report = {
        "class_weights": class_weight_dict,
        "best_production_checkpoint": prod_ckpt.best_metrics,
        "history": {k: v[-8:] for k, v in history_all.items()},
        "warmup_epochs": warmup_epochs,
        "fine_tune_epochs": remaining,
        "sandy_boost": sandy_boost,
    }
    hist_path = ARTIFACTS_DIR / "soil_train_history.json"
    hist_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {hist_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--sandy-boost", type=float, default=1.8)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        sandy_boost=args.sandy_boost,
    )


if __name__ == "__main__":
    main()
