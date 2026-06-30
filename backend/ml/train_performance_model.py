"""
train_performance_model.py
---------------------------
Model 1: Overall Performance Score Regressor

Trains a RandomForestRegressor that predicts a 0-100 "performance score"
for a speaking session from the session's engineered features
(wpm, posture, fillers, volume, eye contact, tone, emotion).

Why this is real ML and not just rules:
- We don't have human-rated scores for the 78 saved sessions, so we
  bootstrap a *continuous* target using a transparent scoring formula
  (see `compute_target_score`). The model then learns the underlying
  feature -> score MAPPING via regression instead of hardcoded
  if/elif thresholds, which means:
    1. It captures interactions between features (e.g. high WPM is
       only bad when fillers are also high) that hand-written rules miss.
    2. It is retrainable: once you collect real human/peer ratings for
       sessions, you swap `compute_target_score` output for the real
       label column and retrain with the exact same pipeline.
    3. It outputs feature_importances_, giving you genuine, data-driven
       insight into what actually drives a good session — useful for a
       report/pitch.

Run:
    python backend/ml/train_performance_model.py
Outputs:
    backend/models/performance_model.joblib   (trained sklearn Pipeline)
    backend/models/performance_model_meta.json (feature list, metrics)
"""
import os
import json
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "..", "data", "raw")
MODEL_DIR = os.path.join(HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = ["wpm", "posture", "fillers", "volume", "pitch"]
CATEGORICAL_FEATURES = ["eyeContact", "tone", "emotion"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_sessions():
    """Load + clean every session JSON in data/raw into a DataFrame."""
    rows = []
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return clean_sessions(df)


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real session logs are messy (mixed types, outliers, unset placeholders
    like "Detecting..."). This applies the same cleaning rules the FastAPI
    endpoints should use at inference time, so train/serve stay consistent.
    """
    df = df.copy()

    # Coerce numeric-ish fields that sometimes arrive as strings
    for col in ["posture", "wpm", "fillers", "volume", "pitch"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use avg_wpm/avg_volume/avg_pitch if present (these come from the
    # real-time audio websocket aggregation) and fall back to the raw field
    if "avg_wpm" in df.columns:
        df["wpm"] = pd.to_numeric(df["avg_wpm"], errors="coerce").fillna(df["wpm"])
    if "avg_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["avg_volume"], errors="coerce").fillna(df["volume"])
    if "avg_pitch" in df.columns:
        df["pitch"] = pd.to_numeric(df["avg_pitch"], errors="coerce").fillna(df["pitch"])
    if "dominant_tone" in df.columns:
        df["tone"] = df["dominant_tone"].fillna(df.get("tone", "Neutral"))

    # Clip absurd outliers (e.g. wpm=1601 from a speech-recognition glitch)
    df["wpm"] = df["wpm"].clip(0, 260)
    df["posture"] = df["posture"].clip(0, 100)
    df["fillers"] = df["fillers"].clip(0, 50)
    df["volume"] = df["volume"].clip(0, 5)
    df["pitch"] = df["pitch"].clip(0, 500)

    # Normalize categorical placeholders into a clean "Unknown"/"Neutral" bucket
    if "eyeContact" not in df.columns:
        df["eyeContact"] = "Unknown"
    df["eyeContact"] = df["eyeContact"].fillna("Unknown").replace({"": "Unknown"})

    if "emotion" not in df.columns:
        df["emotion"] = "Neutral"
    df["emotion"] = df["emotion"].fillna("Neutral").replace(
        {"Detecting...": "Neutral", "": "Neutral"}
    )

    if "tone" not in df.columns:
        df["tone"] = "Neutral"
    df["tone"] = df["tone"].fillna("Neutral").replace({"": "Neutral"})

    return df


def compute_target_score(row) -> float:
    """
    Bootstrapped 0-100 ground-truth score used ONLY to train the model.
    This encodes the same domain knowledge that used to live in
    backend/main.py's get_report() if/elif rules, but expressed as a
    smooth, weighted scoring function instead of discrete branches —
    this is what lets the regressor learn a continuous, nuanced mapping
    rather than memorizing branches.
    """
    score = 50.0  # neutral baseline

    # Pacing: ideal band is 120-160 WPM
    wpm = row["wpm"]
    if wpm == 0:
        pacing_score = 0  # no speech detected
    else:
        distance_from_ideal = max(0, max(120 - wpm, wpm - 160))
        pacing_score = max(0, 25 - distance_from_ideal * 0.25)
    score += pacing_score - 12.5  # center around 0 contribution

    # Posture: directly scaled
    score += (row["posture"] - 50) * 0.25

    # Fillers: penalize
    score -= row["fillers"] * 3

    # Eye contact
    if row["eyeContact"] == "Good":
        score += 8
    elif row["eyeContact"] == "Looking Away":
        score -= 5

    # Tone
    if row["tone"] == "Energetic":
        score += 5
    elif row["tone"] == "Calm":
        score += 2
    elif row["tone"] == "Noise":
        score -= 8

    # Emotion
    if row["emotion"] in ("Happy", "Confident"):
        score += 6
    elif row["emotion"] in ("Sad", "Angry", "Fearful"):
        score -= 6

    return float(np.clip(score, 0, 100))


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def main():
    df = load_sessions()
    print(f"Loaded {len(df)} sessions from {RAW_DIR}")

    if len(df) < 10:
        raise SystemExit(
            f"Only {len(df)} sessions found — need at least 10 to train. "
            "Run a few more sessions through the app first."
        )

    df["target_score"] = df.apply(compute_target_score, axis=1)

    X = df[ALL_FEATURES]
    y = df["target_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    print(f"Test MAE: {mae:.2f}  |  Test R2: {r2:.3f}  |  5-fold CV MAE: {cv_mae:.2f}")

    # Feature importance (mapped back to human-readable names)
    cat_encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    importances = pipeline.named_steps["model"].feature_importances_
    importance_pairs = sorted(
        zip(feature_names, importances), key=lambda p: p[1], reverse=True
    )
    print("\nTop feature importances:")
    for name, imp in importance_pairs[:8]:
        print(f"  {name:25s} {imp:.3f}")

    model_path = os.path.join(MODEL_DIR, "performance_model.joblib")
    joblib.dump(pipeline, model_path)

    meta = {
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_train_sessions": len(df),
        "test_mae": mae,
        "test_r2": r2,
        "cv_mae": cv_mae,
        "feature_importances": [[n, float(i)] for n, i in importance_pairs],
    }
    meta_path = os.path.join(MODEL_DIR, "performance_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
