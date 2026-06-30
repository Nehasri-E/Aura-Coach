# Aura Coach

Aura Coach is a real-time public-speaking practice tool. It watches you through your webcam and mic while you speak, then gives live feedback on posture, eye contact, pace, filler words, volume, pitch, and vocal tone — plus a post-session report with an ML-generated performance score.

## How it works

- **Frontend (React)** — captures webcam/mic in the browser, runs body-language tracking with MediaPipe Pose and facial-expression detection with face-api.js, streams audio to the backend over a WebSocket, and renders live tips during the session plus a report afterward.
- **Backend (FastAPI)** — receives the audio stream, extracts acoustic features (loudness, pitch, spectral centroid, zero-crossing rate via `librosa`), classifies vocal tone, logs session data, and serves a performance report.
- **ML models (scikit-learn)** — two trained RandomForest models replace the original rule-based thresholds:
  - A **performance regressor** that scores a session 0–100 from features like WPM, posture, fillers, volume, pitch, eye contact, tone, and emotion.
  - A **tone classifier** (`Calm` / `Balanced` / `Energetic` / `Noise`) from acoustic features.

  See `README_ML.md` for details on how these were trained and how to retrain them.

## Project structure

\`\`\`
aura-coach-fixed/
├── backend/
│   ├── main.py              # FastAPI app: WebSocket audio stream, session/report endpoints
│   ├── model_loader.py      # Loads the .joblib models at startup
│   ├── ml/                  # Training scripts + audio feature extraction
│   ├── models/              # Trained model artifacts (.joblib + metadata)
│   └── requirements.txt
├── src/
│   ├── pages/                # HomePage, TrainingPage (live session), ReportPage, SessionHistoryPage
│   └── App.js
├── public/models/             # face-api.js model weights (served statically)
├── data/raw/                  # Saved session JSONs (gitignored — not included in version control)
├── dockerfile
└── package.json
\`\`\`

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (project was last run on 3.13)
- A webcam and microphone

## Setup

**1. Clone and configure environment variables**

\`\`\`bash
cp .env.example .env
\`\`\`

The default points the frontend at a local backend (`http://localhost:8010`) — no changes needed for local dev.

**2. Backend**

\`\`\`bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: .\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
\`\`\`

Trained model files are already included in `backend/models/`. To retrain them yourself (e.g. after collecting more sessions), see `README_ML.md`.

**3. Frontend**

\`\`\`bash
npm install
\`\`\`

## Running locally

Two terminals, both servers running at the same time:

\`\`\`bash
# Terminal 1 — backend
cd backend
uvicorn main:app --reload --port 8010

# Terminal 2 — frontend
npm start
\`\`\`

Open `http://localhost:3000`. Click **Start Session**, allow camera/mic access, and the live dashboard will populate.

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/audio-stream` | WebSocket | Streams mic audio, returns live tone/volume/pitch predictions |
| `/api/session/log` | POST | Logs a completed session's data |
| `/api/report/{session_id}` | GET | Returns the full report for a session, including the ML performance score |
| `/api/predict/performance` | POST | Scores an arbitrary feature set (useful for testing / what-if checks) |
| `/api/sessions` | GET | Lists saved sessions |
| `/api/models/status` | GET | Health check confirming both ML models loaded |

## Docker

\`\`\`bash
docker build -t aura-coach .
docker run -p 8080:8080 aura-coach
\`\`\`

The image build trains both models from `data/raw/`. If that folder has fewer than 10 sessions, the build will fail on the training step — either seed it with sample sessions first, or train locally and commit the `.joblib` files in `backend/models/` instead.

## Notes

- `data/raw/` (saved practice sessions) is gitignored by default since it's personal session data — remove that line from `.gitignore` if you want to version it.
- CORS in `backend/main.py` is currently set up for `localhost:3000` and `*.devtunnels.ms`; update `allow_origins` / `allow_origin_regex` if you deploy elsewhere.