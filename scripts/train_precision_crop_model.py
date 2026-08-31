#!/usr/bin/env python3
"""
Train lean precision yield model on Environmental Factors + weather-derived fields.

Features = sensor/app soil (pH, NPK, moisture, texture) + weather (temp, RH, wind,
precip proxy, Tmax/Tmin, GDD, ET0). No history/economic/micronutrient defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ARTIFACTS_DIR, ENV_DATASET_PATH
from app.features.schema import NUMERIC_FEATURE_KEYS
from app.features.vector import estimate_et0, estimate_gdd
from app.knowledge.environmental_maps import DATASET_CROP_TO_ID, DATASET_SOIL_TO_TEXTURE
from app.services.precision_recommender import PRECISION_META_PATH, PRECISION_MODEL_PATH

CAT_COLS = ["soil_texture", "crop_id"]
NUM_COLS = list(NUMERIC_FEATURE_KEYS)


def build_training_frame(max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    if not ENV_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {ENV_DATASET_PATH}")
    df = pd.read_csv(ENV_DATASET_PATH)
    df = df.dropna(subset=["Crop_Type", "Soil_Type", "Crop_Yield"])
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=seed)

    rng = np.random.default_rng(seed)
    rows = []
    for _, row in df.iterrows():
        soil_texture = DATASET_SOIL_TO_TEXTURE.get(row["Soil_Type"], "loamy")
        crop_id = DATASET_CROP_TO_ID.get(row["Crop_Type"])
        if not crop_id:
            continue
        temp = float(row["Temperature"])
        humidity = float(row["Humidity"])
        wind = max(0.0, float(row["Wind_Speed"]))
        # Small physical noise around observed temp for Tmax/Tmin (like a short forecast)
        tmax = temp + float(rng.uniform(2.0, 6.0))
        tmin = temp - float(rng.uniform(2.0, 6.0))
        # Moisture proxy from humidity + soil quality signal in dataset
        sq = float(row.get("Soil_Quality", 40.0))
        moisture = float(np.clip(0.45 * humidity + 0.35 * sq + rng.normal(0, 3), 8, 85))
        # Precip proxy consistent with humid climates in the sheet
        precip = float(np.clip((humidity / 100.0) * 900.0 + rng.normal(0, 40), 50, 2000))

        rows.append(
            {
                "soil_ph": float(row["Soil_pH"]),
                "nitrogen": float(row["N"]),
                "phosphorus": float(row["P"]),
                "potassium": float(row["K"]),
                "soil_moisture_vwc": moisture,
                "precip_accum_mm": precip,
                "temperature_c": temp,
                "temp_max_c": tmax,
                "temp_min_c": tmin,
                "relative_humidity": humidity,
                "wind_speed": wind,
                "gdd": estimate_gdd(temp, tmax, tmin, days=5.0),
                "et0_mm": estimate_et0(temp, humidity, wind),
                "soil_texture": soil_texture,
                "crop_id": crop_id,
                "target_yield": max(0.0, float(row["Crop_Yield"])),
            }
        )
    return pd.DataFrame(rows)


def train(
    max_rows: int | None = None,
    n_estimators: int = 400,
    seed: int = 42,
    model: str = "gbr",
) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = build_training_frame(max_rows=max_rows, seed=seed)
    X = data[NUM_COLS + CAT_COLS]
    y = data["target_yield"].to_numpy()

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    if model == "rf":
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=22,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            random_state=seed,
        )

    pipe = Pipeline([("pre", pre), ("model", regressor)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "n_rows": int(len(data)),
        "n_features": len(NUM_COLS) + len(CAT_COLS),
        "numeric_features": NUM_COLS,
        "categorical_features": CAT_COLS,
        "domains": ["soil", "weather"],
        "model": model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "yield_r2": float(r2_score(y_test, pred)),
        "yield_mae": float(mean_absolute_error(y_test, pred)),
        "baseline_mae": float(
            mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
        ),
    }
    joblib.dump(pipe, PRECISION_MODEL_PATH)
    PRECISION_META_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {PRECISION_MODEL_PATH}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["gbr", "rf"], default="gbr")
    args = parser.parse_args()
    train(
        max_rows=args.max_rows,
        n_estimators=args.n_estimators,
        seed=args.seed,
        model=args.model,
    )


if __name__ == "__main__":
    main()
