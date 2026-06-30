# Aura Coach

A real-time public-speaking coach. Turn on your camera and mic, start talking, and Aura Coach watches your body language and listens to your voice — live — then scores the whole session with a trained ML model at the end.

## What the dashboard actually shows

The live training screen is a three-panel layout: your camera feed in the center, with two stat panels on either side that update continuously while you speak.

**Body Language Analysis** (left panel)
- **Posture Score** — a 0–100% score from MediaPipe Pose landmark tracking
- **Head Tilt** — current head angle in degrees
- **Eye Contact** — Good / Needs Work, from gaze direction relative to the camera

**Speaking Style Feedback** (right panel)
- **Speaking Pace** — live words-per-minute
- **Filler Words** — running count of "um," "uh," "like," etc.
- **Emotion** — facial expression read via face-api.js (e.g. happy, neutral, nervous)
- **Tone** — vocal tone classified live as Calm / Balanced / Energetic / Noise
- **Volume** — current loudness (RMS)
- **Pitch** — current vocal pitch

**Live Tips** (below the camera) — short, real-time nudges generated from your current metrics, e.g. "Speak louder," "Slow down," "Try to look at the camera more."

After you end a session, the **Report page** shows a summary, rule-based insights/recommendations, and an **ML Performance Score (0–100)** from the trained regressor — not just an average of the raw numbers, but a model that's learned how these metrics interact (e.g. fast pace only hurts the score when filler words are also high).

There's also a **Session History** page listing all past sessions so you can track whether posture, pace, fillers, etc. are trending in the right direction over time.

## How it works under the hood

- **Frontend (React)** — captures webcam/mic, runs MediaPipe Pose for body tracking and face-api.js for facial expression, streams audio to the backend over a WebSocket, and renders the live dashboard + report.
- **Backend (FastAPI)** — receives the audio stream, extracts acoustic features (loudness, pitch, spectral centroid, zero-crossing rate via `librosa`), runs the tone classifier on each chunk in real time, logs session data, and builds the post-session report.
- **ML models (scikit-learn, RandomForest)** — trained on 78 real recorded sessions, replacing what used to be hand-written if/else thresholds:

  | Model | Task | Test MAE / Accuracy | Test R² / CV |
  |---|---|---|---|
  | Performance Regressor | 0–100 session score from wpm, posture, fillers, volume, pitch, eyeContact, tone, emotion | ~1.9 pts | R² ~0.96, 5-fold CV MAE ~3.1 |
  | Tone Classifier | Calm / Balanced / Energetic / Noise from 5 acoustic features | ~98% accuracy | 5-fold CV ~99.6% |

  On the performance model, **posture (54%) and speaking pace (19%) are the biggest drivers** of the score, per its feature importances — eye contact comes next. Full training details, why these are real learned models rather than thresholds, and retraining instructions are in `README_ML.md`.

## Project structure

```
aura-coach-fixed/
├── backend/
│   ├── main.py              # FastAPI app: WebSocket audio stream, session/report endpoints
│   ├── model_loader.py      # Loads the .joblib models at startup
│   ├── ml/                  # Training scripts + audio feature extraction
│   ├── models/              # Trained model artifacts (.joblib + metadata)
│   └── requirements.txt
├── src/
│   ├── pages/
│   │   ├── HomePage/           # Landing page
│   │   ├── TrainingPage/       # Live session: camera, dashboard, live tips
│   │   ├── ReportPage/         # Post-session summary + ML score
│   │   └── SessionHistoryPage/ # Past sessions list
│   └── App.js
├── public/models/             # face-api.js model weights (served statically)
├── data/raw/                  # Saved session JSONs (gitignored)
├── dockerfile
└── package.json
```

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (last run on 3.13)
- A webcam and microphone

## Setup

**1. Environment variables**

```bash
cp .env.example .env
```
Default points the frontend at `http://localhost:8010` — fine for local dev as-is.

**2. Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Trained models are already in `backend/models/`. To retrain (e.g. after collecting more sessions), see `README_ML.md`.

**3. Frontend**

```bash
npm install
```

## Running locally

Two terminals, both running at the same time:

```bash
# Terminal 1 — backend
cd backend
uvicorn main:app --reload --port 8010

# Terminal 2 — frontend
npm start
```

Open `http://localhost:3000`, click **Start Session**, allow camera/mic access.

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/audio-stream` | WebSocket | Streams mic audio, returns live tone/volume/pitch predictions |
| `/api/session/log` | POST | Logs a completed session's data |
| `/api/report/{session_id}` | GET | Full report for a session, including the ML performance score |
| `/api/predict/performance` | POST | Score an arbitrary feature set (testing / what-if) |
| `/api/sessions` | GET | List saved sessions |
| `/api/models/status` | GET | Health check confirming both ML models loaded |

## Docker

```bash
docker build -t aura-coach .
docker run -p 8080:8080 aura-coach
```

The build trains both models from `data/raw/`. Fewer than 10 sessions there will fail the build — seed sample sessions first, or train locally and commit the `.joblib` files in `backend/models/` instead.

## Notes

- `data/raw/` is gitignored by default since it's personal session data.
- CORS in `backend/main.py` currently allows `localhost:3000` and `*.devtunnels.ms`; update `allow_origins` / `allow_origin_regex` for other deployments.