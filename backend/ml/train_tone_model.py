"""
train_tone_model.py
---------------------
Model 2: Vocal Tone Classifier (Calm / Balanced / Energetic / Noise)

Why this replaces the old hardcoded logic:
  backend/main.py used to decide tone with a single hand-written rule:
      tone = "Calm" if rms < 0.02 else "Balanced" if rms < 0.05 else "Energetic"
  That's a 1-feature, 2-threshold rule. This script trains a small
  RandomForestClassifier on FOUR audio features extracted with librosa
  (RMS loudness, pitch mean, pitch variability, spectral centroid), so
  the model can pick up patterns a single threshold can't, e.g. "loud but
  monotone" vs "moderate volume but highly variable pitch".

Why synthetic training audio:
  The sessions saved so far never persisted raw audio (only derived
  scalar features), and there's no public dataset wired in yet. So this
  script SYNTHESIZES short waveforms with the acoustic *characteristics*
  associated with each tone label (loudness envelope + pitch range +
  jitter), extracts the same librosa features the live FastAPI endpoint
  will compute, and trains on those. This keeps train/serve feature
  extraction identical (see backend/ml/audio_features.py), so swapping in
  real labeled mic recordings later is a drop-in replacement: just point
  `build_training_set()` at a folder of labeled .wav files instead.

Run:
    python backend/ml/train_tone_model.py
Outputs:
    backend/models/tone_model.joblib
    backend/models/tone_model_meta.json
"""
import os
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    from audio_features import extract_features_from_waveform, FEATURE_NAMES
except ImportError:
    from ml.audio_features import extract_features_from_waveform, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SR = 16000
DURATION = 1.5  # seconds per synthetic clip
N_SAMPLES_PER_CLASS = 120

LABELS = ["Noise", "Calm", "Balanced", "Energetic"]


def synth_clip(label: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one synthetic waveform whose acoustic character matches the
    given tone label. Each label gets a distinct loudness range, pitch
    range, pitch jitter and harmonic richness, mirroring how a real human
    voice differs across these speaking styles.
    """
    n = int(SR * DURATION)
    t = np.linspace(0, DURATION, n, endpoint=False)

    if label == "Noise":
        # broadband noise, no real pitch structure, low-to-mid energy
        amp = rng.uniform(0.005, 0.03)
        y = rng.normal(0, 1, n) * amp
        return y.astype(np.float32)

    if label == "Calm":
        f0 = rng.uniform(90, 140)      # lower, steady pitch
        jitter = rng.uniform(0.0, 0.02)
        amp = rng.uniform(0.01, 0.05)
        n_harmonics = 3
    elif label == "Balanced":
        f0 = rng.uniform(130, 190)
        jitter = rng.uniform(0.02, 0.06)
        amp = rng.uniform(0.05, 0.12)
        n_harmonics = 4
    else:  # Energetic
        f0 = rng.uniform(170, 260)     # higher, more variable pitch
        jitter = rng.uniform(0.05, 0.12)
        amp = rng.uniform(0.12, 0.28)
        n_harmonics = 6

    # Pitch contour with jitter (natural micro-variation in voice)
    pitch_contour = f0 * (1 + jitter * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t))
    phase = 2 * np.pi * np.cumsum(pitch_contour) / SR

    y = np.zeros(n)
    for h in range(1, n_harmonics + 1):
        y += (1.0 / h) * np.sin(h * phase)

    # Amplitude envelope (speech-like, not constant tone)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1.5, 4) * t + rng.uniform(0, 6))
    y = y * envelope * amp

    # A touch of background noise for realism
    y += rng.normal(0, amp * 0.05, n)

    return y.astype(np.float32)


def build_training_set():
    rng = np.random.default_rng(42)
    X, y = [], []
    for label in LABELS:
        for _ in range(N_SAMPLES_PER_CLASS):
            wav = synth_clip(label, rng)
            feats = extract_features_from_waveform(wav, SR)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)


def main():
    print("Synthesizing training audio + extracting librosa features...")
    X, y = build_training_set()
    print(f"Built {len(y)} samples across {len(LABELS)} classes: {LABELS}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    importance_pairs = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda p: p[1], reverse=True
    )
    print("\nFeature importances:")
    for name, imp in importance_pairs:
        print(f"  {name:20s} {imp:.3f}")

    model_path = os.path.join(MODEL_DIR, "tone_model.joblib")
    joblib.dump(clf, model_path)

    meta = {
        "labels": LABELS,
        "feature_names": FEATURE_NAMES,
        "n_train_samples": len(y_train),
        "test_accuracy": acc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "feature_importances": [[n, float(i)] for n, i in importance_pairs],
        "note": (
            "Trained on synthetic audio matching each tone's acoustic profile "
            "(loudness, pitch range, pitch jitter). Swap build_training_set() "
            "for real labeled recordings when available."
        ),
    }
    meta_path = os.path.join(MODEL_DIR, "tone_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
