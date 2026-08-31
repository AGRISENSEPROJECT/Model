"""Compare ML models for crop yield ranking — run offline, not at request time."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ARTIFACTS_DIR
from app.features.schema import NUMERIC_FEATURE_KEYS
from scripts.train_precision_crop_model import CAT_COLS, NUM_COLS, build_training_frame


def _build_pipeline(model_name: str, seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    if model_name == "rf":
        reg = RandomForestRegressor(n_estimators=300, max_depth=22, random_state=seed, n_jobs=-1)
    elif model_name == "svr":
        reg = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    else:
        reg = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            random_state=seed,
        )
    return Pipeline([("pre", pre), ("model", reg)])


def compare_models(seed: int = 42) -> dict:
    data = build_training_frame(seed=seed)
    X = data[NUM_COLS + CAT_COLS]
    y = data["target_yield"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    results = []
    for name in ("gbr", "rf", "svr"):
        pipe = _build_pipeline(name, seed)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(
            {
                "model": name,
                "yield_r2": float(r2_score(y_test, pred)),
                "yield_mae": float(mean_absolute_error(y_test, pred)),
            }
        )

    best = max(results, key=lambda r: r["yield_r2"])
    out = {
        "dataset_rows": len(data),
        "feature_count": len(NUMERIC_FEATURE_KEYS) + len(CAT_COLS),
        "comparison": results,
        "recommended_model": best["model"],
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "model_comparison_report.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compare_models(seed=args.seed)


if __name__ == "__main__":
    main()
