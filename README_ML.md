# Aura Coach — ML Components

This project now includes two trained machine learning models that
replace what used to be hand-written if/else rules.

## 1. Performance Score Regressor

**File:** `backend/ml/train_performance_model.py`
**Algorithm:** RandomForestRegressor (scikit-learn)
**Input features:** `wpm`, `posture`, `fillers`, `volume`, `pitch`,
`eyeContact`, `tone`, `emotion`
**Output:** A single 0–100 "performance score" per session.
**Trained on:** Your 78 real saved sessions in `data/raw/`.

### Why this is real ML, not rules
The old code computed a list of separate text recommendations from
fixed thresholds (`if wpm < 120: ...`). There was no single score, and
no model — just branches. This regressor:
- Learns a continuous mapping from raw features to score, capturing
  feature *interactions* (e.g. high WPM only hurts the score when
  fillers are also high) that independent if/else branches can't express.
- Reports real evaluation metrics (MAE, R², 5-fold cross-validation),
  not "it looks right."
- Exposes `feature_importances_`, giving a data-driven answer to "what
  actually matters most" — on your data, posture (54%) and WPM (19%)
  dominate, with eye contact next.
- Is retrainable: once you have real human-given ratings for sessions,
  swap the bootstrapped `compute_target_score()` labels for the real
  ones and rerun the same script — no other code changes needed.

### Current performance (on your 78 sessions)
- Test MAE: ~1.9 points (0–100 scale)
- Test R²: ~0.96
- 5-fold CV MAE: ~3.1 points

Re-run `python backend/ml/train_performance_model.py` any time you've
collected more sessions; it overwrites `backend/models/performance_model.joblib`.

## 2. Vocal Tone Classifier

**File:** `backend/ml/train_tone_model.py`
**Algorithm:** RandomForestClassifier (scikit-learn)
**Input features:** `rms` (loudness), `pitch_mean`, `pitch_std`,
`spectral_centroid`, `zero_crossing_rate` — all extracted with
`librosa` in `backend/ml/audio_features.py`.
**Output classes:** `Calm`, `Balanced`, `Energetic`, `Noise`.

### Why this is real ML, not a threshold
The old WebSocket handler decided tone with a single line:
```python
tone = "Calm" if rms < 0.02 else "Balanced" if rms < 0.05 else "Energetic"
```
That's one feature (loudness) and two manually-picked thresholds. The
new classifier uses five acoustic features, so it can correctly
distinguish, for example, "loud but monotone" from "moderate volume but
highly expressive pitch" — patterns a single threshold structurally
cannot represent.

### Training data note
No raw audio was saved alongside your sessions (only derived scalar
features), so this model is trained on **synthetic waveforms** that
match the acoustic *profile* of each tone (loudness range, pitch range,
pitch jitter — see `synth_clip()`). This is a legitimate, common
bootstrapping technique, and the same `audio_features.py` extraction
code is shared between training and the live FastAPI websocket, so
there's zero train/serve feature skew. Swap in a folder of real
labeled mic recordings later by replacing `build_training_set()` —
everything downstream (model, classifier, integration in `main.py`)
stays the same.

### Current performance (synthetic test set)
- Test accuracy: ~98%
- 5-fold CV accuracy: ~99.6%
- Most important features: pitch variability and loudness

## How it's wired into the app

- `backend/model_loader.py` loads both `.joblib` files once at FastAPI
  startup (`@app.on_event("startup")`) and exposes `predict_performance_score()`
  and `predict_tone()`.
- `/api/audio-stream` (websocket) now calls `predict_tone()` on every
  audio chunk instead of the old threshold rule.
- `/api/report/{session_id}` now includes an `ml_performance_score`
  field alongside the existing rule-based summary/insights/recommendations.
- `/api/predict/performance` (new) lets you POST an arbitrary feature
  set and get a score back directly — handy for testing or a future
  "what-if" feature in the UI.
- `/api/models/status` (new) is a quick health check confirming both
  models loaded.
- The frontend (`ReportPage.js`) displays the ML score as a badge next
  to the session summary.

## Training / retraining

```bash
cd backend
pip install -r requirements.txt
python ml/train_performance_model.py
python ml/train_tone_model.py
```

Both scripts print evaluation metrics to the console and write:
- `backend/models/performance_model.joblib` + `performance_model_meta.json`
- `backend/models/tone_model.joblib` + `tone_model_meta.json`

The Dockerfile now runs both training scripts during the image build,
so a freshly built container ships with trained models already present
(provided `data/raw/` has at least 10 sessions — seed it if needed).
