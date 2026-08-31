#!/usr/bin/env python3
"""Train MobileNetV2 soil texture classifier with class weights + saved index map."""
from __future__ import annotations

import argparse
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


def train(epochs: int = 20) -> None:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "validation"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training directory: {train_dir}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
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
    print("Class weights:", class_weight_dict)

    n_classes = len(train_gen.class_indices)
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(SOIL_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )
    model.save(SOIL_MODEL_PATH)
    print(f"Saved soil model → {SOIL_MODEL_PATH}")
    print("Final val accuracy:", history.history.get("val_accuracy", [None])[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
