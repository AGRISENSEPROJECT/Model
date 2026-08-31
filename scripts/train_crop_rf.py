#!/usr/bin/env python3
"""Train RandomForest crop recommender with the production feature pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ARTIFACTS_DIR, CROP_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH
from app.knowledge.crops import CROP_SUITABILITY


def generate_synthetic_data(samples_per_crop: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for crop, reqs in CROP_SUITABILITY.items():
        for _ in range(samples_per_crop):
            # 70% in-range samples, 30% near-boundary noise for robustness
            if rng.random() < 0.7:
                temperature = rng.uniform(reqs["temperature"]["min"], reqs["temperature"]["max"])
                humidity = rng.uniform(reqs["humidity"]["min"], reqs["humidity"]["max"])
                rainfall = rng.uniform(reqs["rainfall"]["min"], reqs["rainfall"]["max"])
                soil_texture = rng.choice(reqs["soil_texture"])
            else:
                temperature = rng.uniform(
                    reqs["temperature"]["min"] - 5, reqs["temperature"]["max"] + 5
                )
                humidity = rng.uniform(reqs["humidity"]["min"] - 10, reqs["humidity"]["max"] + 10)
                rainfall = rng.uniform(reqs["rainfall"]["min"] - 100, reqs["rainfall"]["max"] + 100)
                soil_texture = rng.choice(["sandy", "loamy", "clayey", "alluvial"])
            rows.append([soil_texture, temperature, humidity, rainfall, crop])
    return pd.DataFrame(
        rows, columns=["soil_texture", "temperature", "humidity", "rainfall", "crop"]
    )


def train(samples_per_crop: int = 2000, seed: int = 42) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(samples_per_crop=samples_per_crop, seed=seed)

    label_encoder = LabelEncoder()
    df["soil_texture_code"] = label_encoder.fit_transform(df["soil_texture"])

    scaler = StandardScaler()
    climate = df[["temperature", "humidity", "rainfall"]].to_numpy(dtype=float)
    scaled = scaler.fit_transform(climate)
    X = np.column_stack([df["soil_texture_code"].to_numpy(), scaled])
    y = df["crop"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, CROP_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved model → {CROP_MODEL_PATH}")
    print(f"Saved scaler → {SCALER_PATH}")
    print(f"Saved label encoder → {LABEL_ENCODER_PATH}")
    print("Feature order: [soil_texture_code, scaled_temp, scaled_humidity, scaled_rainfall]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-crop", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(samples_per_crop=args.samples_per_crop, seed=args.seed)


if __name__ == "__main__":
    main()
