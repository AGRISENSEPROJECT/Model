"""Soil texture inference with Keras alphabetical class-index correction."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config import (
    DEFAULT_SOIL_CLASS_INDICES,
    IMG_SIZE,
    SOIL_CLASS_INDICES_PATH,
    SOIL_MODEL_PATH,
)

_model = None
_model_lock = threading.Lock()
_index_to_label: dict[int, str] | None = None


def load_class_indices() -> dict[str, int]:
    if SOIL_CLASS_INDICES_PATH.exists():
        with open(SOIL_CLASS_INDICES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): int(v) for k, v in data.items()}
    return dict(DEFAULT_SOIL_CLASS_INDICES)


def index_to_label_map() -> dict[int, str]:
    global _index_to_label
    if _index_to_label is None:
        indices = load_class_indices()
        _index_to_label = {idx: name for name, idx in indices.items()}
    return _index_to_label


def save_class_indices(class_indices: dict[str, int], path: Path | None = None) -> Path:
    target = path or SOIL_CLASS_INDICES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(class_indices.items(), key=lambda kv: kv[1])), fh, indent=2)
    global _index_to_label
    _index_to_label = {int(v): str(k) for k, v in class_indices.items()}
    return target


def get_soil_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not SOIL_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Soil model missing at {SOIL_MODEL_PATH}. Run scripts/train_soil_texture.py"
                )
            from tensorflow.keras.models import load_model

            _model = load_model(SOIL_MODEL_PATH)
    return _model


def preprocess_image(image_path: str | Path) -> np.ndarray:
    from tensorflow.keras.utils import img_to_array, load_img

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_texture(image_path: str | Path) -> dict[str, Any]:
    """Return texture label + confidence using corrected Keras index mapping."""
    model = get_soil_model()
    batch = preprocess_image(image_path)
    probs = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    mapping = index_to_label_map()
    if predicted_idx not in mapping:
        raise ValueError(
            f"Model predicted index {predicted_idx} but class map is {mapping}. "
            "Retrain or regenerate artifacts/soil_class_indices.json"
        )
    texture = mapping[predicted_idx]
    confidence = float(probs[predicted_idx])
    per_class = {mapping[i]: float(probs[i]) for i in range(len(probs)) if i in mapping}
    return {
        "texture": texture,
        "confidence": round(confidence, 4),
        "probabilities": per_class,
        "class_index": predicted_idx,
        "class_index_source": str(SOIL_CLASS_INDICES_PATH.name)
        if SOIL_CLASS_INDICES_PATH.exists()
        else "default_alphabetical",
    }
