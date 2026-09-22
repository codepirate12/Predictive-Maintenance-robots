"""
Training script for the Predictive Maintenance model (AI4I 2020 dataset).

Run from the project root:
    python -m backend.train_model

What this does:
1. Loads data/ai4i2020.csv.
2. One-hot encodes `Type` (L/M/H) the same way model_service.py expects:
   Type_L, Type_M columns, with H as the implicit baseline.
3. Splits into train/test BEFORE any resampling, stratified on
   `Machine failure`, so the test set keeps the real ~3.4% failure rate.
4. Applies SMOTE to the TRAINING split only (backend/smote.py) to correct
   the class imbalance (339 failures out of 10,000 rows).
5. Trains a RandomForestClassifier on the balanced training data.
6. Evaluates on the untouched, imbalanced test split -- this is the
   number that actually reflects real-world performance.
7. Saves the model to models/predictive_maintenance_model.pkl and the
   real metrics to models/performance_metrics.json, which
   model_service.py now serves instead of hardcoded numbers.

Known limitation (flagged, not silently fixed):
   TWF/HDF/PWF/OSF/RNF are the individual fault flags that make up the
   `Machine failure` label (failure = 1 iff any flag = 1). They are kept
   as input features here ONLY to match the existing PredictionInput
   schema / frontend contract. This does mean the model can partly
   "cheat" by reading the flags directly rather than the raw sensor
   values. If in production a prediction request is only ever made
   BEFORE any fault flag is known (all flags = 0), consider retraining
   without those five columns for a more honest sensor-only model.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from backend.smote import smote_oversample

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Type_L",
    "Type_M",
]
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42

DATA_PATH_CANDIDATES = ["data/ai4i2020.csv", "ai4i2020.csv"]
MODEL_OUT_PATH = "models/predictive_maintenance_model.pkl"
METRICS_OUT_PATH = "models/performance_metrics.json"


def _load_dataset() -> pd.DataFrame:
    path = next((p for p in DATA_PATH_CANDIDATES if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(
            f"Could not find ai4i2020.csv in any of: {DATA_PATH_CANDIDATES}"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)
    # drop_first alphabetically drops 'H' (H < L < M), leaving Type_L, Type_M
    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0

    X = encoded[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def train() -> None:
    df = _load_dataset()
    X, y = _build_features(df)

    print(f"Class balance before split -> failures: {y.sum()} / {len(y)} "
          f"({100 * y.mean():.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --- Resample ONLY the training split ---
    X_train_bal, y_train_bal = smote_oversample(
        X_train.values,
        y_train.values,
        minority_label=1,
        k_neighbors=5,
        target_ratio=1.0,
        random_state=RANDOM_STATE,
    )
    print(f"Training set after SMOTE: {len(y_train_bal):,} rows "
          f"({int(y_train_bal.sum()):,} positive / "
          f"{int(len(y_train_bal) - y_train_bal.sum()):,} negative)")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(pd.DataFrame(X_train_bal, columns=FEATURE_COLUMNS), y_train_bal)

    # --- Evaluate on the REAL, untouched, imbalanced test split ---
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_set_size": int(len(y_test)),
        "test_set_failure_count": int(y_test.sum()),
        "training_method": "SMOTE-balanced training split; evaluated on untouched imbalanced test split",
    }

    feature_importances = sorted(
        [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    metrics["feature_importances"] = feature_importances

    print("\n--- Held-out test set performance (deployed model, matches API schema) ---")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:>10}: {metrics[k]}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT_PATH)
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {MODEL_OUT_PATH}")
    print(f"Saved metrics -> {METRICS_OUT_PATH}")


if __name__ == "__main__":
    train()
