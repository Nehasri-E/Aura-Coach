"""
audio_features.py
-------------------
Shared feature extraction for the tone/emotion classifier.

IMPORTANT: This module is imported by BOTH:
  - backend/ml/train_tone_model.py   (training time)
  - backend/main.py                  (live inference, inside the
                                       /api/audio-stream websocket)

Keeping the exact same feature extraction code in one place guarantees
there's no train/serve skew — a classic, easy-to-miss ML bug where the
model is trained on features computed one way but served on features
computed slightly differently (e.g. different librosa parameters),
silently degrading real-world accuracy.
"""
import numpy as np
import librosa

FEATURE_NAMES = [
    "rms",                  # loudness
    "pitch_mean",           # average fundamental frequency
    "pitch_std",            # pitch variability (monotone vs expressive)
    "spectral_centroid",    # brightness/"sharpness" of the sound
    "zero_crossing_rate",   # noisiness / sibilance proxy
]


def extract_features_from_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a fixed-length feature vector from a raw audio waveform.
    Returns a numpy array in the order defined by FEATURE_NAMES.
    Designed to be cheap enough to run on small (~1-2 second) chunks
    in real time inside the websocket handler.
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 256:
        # too short to extract meaningful spectral features
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    rms = float(np.sqrt(np.mean(y ** 2)))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch_mean = float(np.mean(f0)) if len(f0) else 0.0
        pitch_std = float(np.std(f0)) if len(f0) else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = float(np.mean(centroid))
    except Exception:
        spectral_centroid = 0.0

    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate = float(np.mean(zcr))
    except Exception:
        zero_crossing_rate = 0.0

    return np.array(
        [rms, pitch_mean, pitch_std, spectral_centroid, zero_crossing_rate],
        dtype=np.float32,
    )
