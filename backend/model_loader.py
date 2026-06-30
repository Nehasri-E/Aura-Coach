"""
model_loader.py
-----------------
Loads the two trained ML models once at FastAPI startup and exposes
small prediction helper functions used by backend/main.py.

If a model file is missing (e.g. you haven't run the training scripts
yet), predictions fall back gracefully to None so the rest of the app
keeps working — main.py falls back to the original rule-based logic
in that case.
"""
import os
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

PERFORMANCE_MODEL_PATH = os.path.join(MODEL_DIR, "performance_model.joblib")
TONE_MODEL_PATH = os.path.join(MODEL_DIR, "tone_model.joblib")

_performance_model = None
_tone_model = None


def _safe_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"⚠️ Failed to load model at {path}: {e}")
    else:
        print(f"⚠️ Model file not found: {path} (run the matching train_*.py script)")
    return None


def load_models():
    """Call once at FastAPI startup."""
    global _performance_model, _tone_model
    _performance_model = _safe_load(PERFORMANCE_MODEL_PATH)
    _tone_model = _safe_load(TONE_MODEL_PATH)
    print(f"✅ Performance model loaded: {_performance_model is not None}")
    print(f"✅ Tone model loaded: {_tone_model is not None}")


def predict_performance_score(features: dict):
    """
    features: dict with keys wpm, posture, fillers, volume, pitch,
              eyeContact, tone, emotion
    Returns a float 0-100, or None if the model isn't loaded.
    """
    if _performance_model is None:
        return None
    row = pd.DataFrame([{
        "wpm": features.get("wpm", 0),
        "posture": features.get("posture", 0),
        "fillers": features.get("fillers", 0),
        "volume": features.get("volume", 0),
        "pitch": features.get("pitch", 0),
        "eyeContact": features.get("eyeContact", "Unknown"),
        "tone": features.get("tone", "Neutral"),
        "emotion": features.get("emotion", "Neutral"),
    }])
    try:
        pred = _performance_model.predict(row)[0]
        return float(np.clip(pred, 0, 100))
    except Exception as e:
        print(f"⚠️ performance model prediction failed: {e}")
        return None


def predict_tone(feature_vector: np.ndarray):
    """
    feature_vector: np.ndarray matching audio_features.FEATURE_NAMES order
                     [rms, pitch_mean, pitch_std, spectral_centroid, zcr]
    Returns (label: str, confidence: float) or (None, None) if unavailable.
    """
    if _tone_model is None:
        return None, None
    try:
        proba = _tone_model.predict_proba([feature_vector])[0]
        idx = int(np.argmax(proba))
        label = _tone_model.classes_[idx]
        confidence = float(proba[idx])
        return label, confidence
    except Exception as e:
        print(f"⚠️ tone model prediction failed: {e}")
        return None, None


def models_status():
    return {
        "performance_model_loaded": _performance_model is not None,
        "tone_model_loaded": _tone_model is not None,
    }
