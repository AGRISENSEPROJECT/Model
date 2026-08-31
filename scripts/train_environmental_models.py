#!/usr/bin/env python3
"""Train yield + soil-quality models from Environmental Factors dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ARTIFACTS_DIR, DATA_DIR
from app.knowledge.environmental_maps import DATASET_CROP_TO_ID

ENV_CSV = DATA_DIR / "environmental" / "crop_yield_dataset.csv"
YIELD_MODEL_PATH = ARTIFACTS_DIR / "yield_predictor.pkl"
QUALITY_MODEL_PATH = ARTIFACTS_DIR / "soil_quality_predictor.pkl"
ENV_META_PATH = ARTIFACTS_DIR / "environmental_model_meta.json"

FEATURE_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Soil_pH",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "N",
    "P",
    "K",
]
CAT_COLS = ["Soil_Type", "Crop_Type"]
NUM_COLS = ["Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K"]


def load_frame(path: Path = ENV_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Place crop_yield_dataset.csv under data/environmental/"
        )
    df = pd.read_csv(path)
    required = FEATURE_COLS + ["Crop_Yield", "Soil_Quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    # Clip nonsense wind / keep all yield rows (zeros are informative: crop failure)
    df = df.copy()
    df["Wind_Speed"] = df["Wind_Speed"].clip(lower=0)
    df["crop_id"] = df["Crop_Type"].map(DATASET_CROP_TO_ID)
    return df.dropna(subset=["crop_id"])


def _make_regressor(n_estimators: int, seed: int) -> Pipeline:
    # dense one-hot — sklearn RF does not accept sparse matrices
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("rf", model)])


def train(n_estimators: int = 200, seed: int = 42, test_size: float = 0.2) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    X = df[FEATURE_COLS]
    y_yield = df["Crop_Yield"].to_numpy()
    y_quality = df["Soil_Quality"].to_numpy()

    X_train, X_test, y_tr, y_te, q_tr, q_te = train_test_split(
        X, y_yield, y_quality, test_size=test_size, random_state=seed
    )

    yield_pipe = _make_regressor(n_estimators, seed)
    quality_pipe = _make_regressor(max(100, n_estimators // 2), seed)

    yield_pipe.fit(X_train, y_tr)
    quality_pipe.fit(X_train, q_tr)

    y_pred = yield_pipe.predict(X_test)
    q_pred = quality_pipe.predict(X_test)

    metrics = {
        "n_rows": int(len(df)),
        "crops": sorted(df["Crop_Type"].unique().tolist()),
        "soils": sorted(df["Soil_Type"].unique().tolist()),
        "yield": {
            "r2": float(r2_score(y_te, y_pred)),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "baseline_mae": float(mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean()))),
        },
        "soil_quality": {
            "r2": float(r2_score(q_te, q_pred)),
            "mae": float(mean_absolute_error(q_te, q_pred)),
        },
        "feature_cols": FEATURE_COLS,
    }

    joblib.dump(yield_pipe, YIELD_MODEL_PATH)
    joblib.dump(quality_pipe, QUALITY_MODEL_PATH)
    ENV_META_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved {YIELD_MODEL_PATH}")
    print(f"Saved {QUALITY_MODEL_PATH}")
    print(f"Saved {ENV_META_PATH}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(n_estimators=args.n_estimators, seed=args.seed)


if __name__ == "__main__":
    main()
