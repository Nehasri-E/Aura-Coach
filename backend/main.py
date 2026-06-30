from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os, json, uuid, io
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
from collections import defaultdict

try:
    # Works when run as `uvicorn backend.main:app` from the project root
    from backend.model_loader import load_models, predict_performance_score, predict_tone, models_status
    from backend.ml.audio_features import extract_features_from_waveform
except ImportError:
    # Works when run from inside backend/ (e.g. `uvicorn main:app`)
    from model_loader import load_models, predict_performance_score, predict_tone, models_status
    from ml.audio_features import extract_features_from_waveform


# -------------------------------------------
# 🚀 FASTAPI INITIAL SETUP
# -------------------------------------------
app = FastAPI()


@app.on_event("startup")
async def _load_ml_models():
    # Loads backend/models/performance_model.joblib and tone_model.joblib
    # (run backend/ml/train_performance_model.py and train_tone_model.py
    # once before starting the server if these don't exist yet)
    load_models()


# ✅ Updated CORS setup — supports VS Code dev tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                     # Local React dev
        "https://d32w6h95-3000.inc1.devtunnels.ms",  # Your existing tunnel
    ],
    # ✅ Also allow any *.devtunnels.ms origin, so a fresh tunnel URL
    # (which changes when VS Code regenerates it) keeps working without
    # needing a code change every time.
    allow_origin_regex=r"https://.*\.devtunnels\.ms",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directory for saving logs
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


# Dictionary to store active audio metrics temporarily
SESSION_AUDIO_DATA = defaultdict(lambda: {"volume": [], "pitch": [], "tone": [], "wpm": []})


# -------------------------------------------
# 🟢 Log a new training session (append/merge safe)
# -------------------------------------------
@app.post("/api/session/log")
async def log_session(req: Request):
    """
    Store or update session metrics + audio summaries into a JSON file.
    """
    data = await req.json()
    session_id = data.get("session_id") or f"session_{uuid.uuid4().hex}"
    data["timestamp"] = datetime.now().isoformat()

    # If real-time audio data exists for this session, aggregate it
    if session_id in SESSION_AUDIO_DATA:
        audio_data = SESSION_AUDIO_DATA[session_id]
        data["avg_volume"] = float(np.mean(audio_data["volume"])) if audio_data["volume"] else 0.0
        data["avg_pitch"] = float(np.mean(audio_data["pitch"])) if audio_data["pitch"] else 0.0
        data["avg_wpm"] = float(np.mean(audio_data["wpm"])) if audio_data["wpm"] else float(data.get("wpm", 0))
        if audio_data["tone"]:
            tones = audio_data["tone"]
            data["dominant_tone"] = max(set(tones), key=tones.count)
        else:
            data["dominant_tone"] = "Neutral"
        # clear stored audio after merging
        del SESSION_AUDIO_DATA[session_id]

    file_path = os.path.join(RAW_DIR, f"{session_id}.json")

    # merge updates if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing = json.load(f)
        existing.update(data)
        with open(file_path, "w") as f:
            json.dump(existing, f, indent=4)
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    print(f"✅ Session saved with live audio data → {file_path}")
    return {"status": "ok", "session_id": session_id}


# -------------------------------------------
# 🔊 Real-Time Audio Streaming with Progressive Storage
# -------------------------------------------
@app.websocket("/api/audio-stream")
async def audio_stream(ws: WebSocket):
    """
    Receives small audio chunks from frontend, analyzes live features,
    stores them for session-level aggregation when frontend provides session_id.
    Frontend may first send a JSON text message: {"session_id":"session_xxx"} to bind chunks.
    """
    await ws.accept()
    print("🎙️ WebSocket audio stream connected")
    sr = 16000
    session_id = None

    try:
        while True:
            msg = await ws.receive()

            # If frontend sends JSON control message with session_id
            if "text" in msg and msg["text"]:
                try:
                    control = json.loads(msg["text"])
                    if isinstance(control, dict) and control.get("session_id"):
                        session_id = control.get("session_id")
                        print(f"📝 Session ID bound: {session_id}")
                except Exception:
                    # not JSON control — ignore
                    pass
                continue

            data = msg.get("bytes", None)
            if not data:
                continue

            # Decode PCM16 or WAV
            try:
                y, _ = sf.read(io.BytesIO(data), dtype="float32")
            except Exception:
                try:
                    y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                except Exception:
                    continue

            if len(y) < 100:
                continue

            # --- ML Feature Extraction (shared with backend/ml/train_tone_model.py) ---
            features = extract_features_from_waveform(y, sr)
            rms, pitch, pitch_std, spectral_centroid, zcr = features

            # Basic noise gate: ignore extremely quiet chunks (mic silence)
            if rms < 0.001:
                # still send a small heartbeat so frontend knows server is alive
                await ws.send_json({"volume": round(rms*100,2), "pitch": 0.0, "tone": "Noise", "wpm": 0})
                continue

            # 🧠 ML tone classification (RandomForest trained on rms, pitch
            # mean/std, spectral centroid, zero-crossing rate) instead of
            # the old single-threshold rule. Falls back to the simple rule
            # only if the model file hasn't been trained/loaded yet.
            tone, tone_confidence = predict_tone(features)
            if tone is None:
                tone = "Calm" if rms < 0.02 else "Balanced" if rms < 0.05 else "Energetic"
                tone_confidence = None

            wpm_est = int(120 + (rms * 200))

            # store in-memory for session aggregation
            if session_id:
                SESSION_AUDIO_DATA[session_id]["volume"].append(rms * 100)
                SESSION_AUDIO_DATA[session_id]["pitch"].append(pitch)
                SESSION_AUDIO_DATA[session_id]["tone"].append(tone)
                SESSION_AUDIO_DATA[session_id]["wpm"].append(wpm_est)

            # send live metrics back
            await ws.send_json({
                "volume": round(rms * 100, 2),
                "pitch": round(float(pitch), 1),
                "tone": tone,
                "tone_confidence": round(tone_confidence, 3) if tone_confidence is not None else None,
                "wpm": wpm_est,
            })

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")
    except Exception as e:
        print("⚠️ Error in audio processing:", e)
    finally:
        print("🔴 Audio WebSocket closed.")


# -------------------------------------------
# 🧠 Generate Performance Report
# -------------------------------------------
@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    file_path = os.path.join(RAW_DIR, f"{session_id}.json")

    if not os.path.exists(file_path):
        return {"error": "Session not found"}

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract metrics
    wpm = data.get("avg_wpm", data.get("wpm", 0))
    posture = float(data.get("posture", 0))
    eye_contact = data.get("eyeContact", "Unknown")
    fillers = data.get("fillers", 0)
    emotion = data.get("emotion", "Neutral")
    tone = data.get("dominant_tone", data.get("tone", "Balanced"))
    pitch = round(data.get("avg_pitch", 0), 1)
    volume = round(data.get("avg_volume", 0), 1)

    summary = f"You spoke at {int(wpm)} WPM with {fillers} filler words. Avg volume: {volume}, pitch: {pitch} Hz."

    insights = {
        "posture": f"Your posture score was {posture}%, showing {'strong alignment' if posture > 80 else 'room for improvement'}.",
        "eye_contact": f"Eye contact was {eye_contact.lower()}, indicating {'engagement' if eye_contact == 'Good' else 'inconsistent focus'}.",
        "emotion": f"Dominant facial emotion: {emotion}.",
        "tone": f"Vocal tone was mostly {tone.lower()}.",
    }

    recs = []
    if wpm < 120:
        recs.append("Increase your speaking pace slightly for energy.")
    elif wpm > 160:
        recs.append("Slow down a bit for clarity and emphasis.")
    else:
        recs.append("Pace was balanced and natural.")

    if posture < 70:
        recs.append("Maintain upright shoulders and balanced head alignment.")
    if fillers > 5:
        recs.append("Reduce filler words such as 'um' and 'like' for smoother delivery.")
    if tone.lower() == "calm":
        recs.append("Consider adding energy to sound more engaging.")
    if tone.lower() == "energetic":
        recs.append("Good vocal projection — keep it controlled for clarity.")
    if emotion.lower() in ["sad", "angry", "fearful"]:
        recs.append("A warmer tone and expression can improve connection.")

    if not recs:
        recs.append("Excellent performance overall! Keep refining consistency.")

    # 🧠 ML Performance Score — RandomForestRegressor trained on your past
    # sessions (backend/ml/train_performance_model.py). This is a learned,
    # data-driven score in addition to the rule-based summary/recs above —
    # falls back to None if the model hasn't been trained/loaded yet.
    ml_score = predict_performance_score({
        "wpm": wpm,
        "posture": posture,
        "fillers": fillers,
        "volume": data.get("avg_volume", data.get("volume", 0)),
        "pitch": data.get("avg_pitch", data.get("pitch", 0)),
        "eyeContact": eye_contact,
        "tone": tone,
        "emotion": emotion,
    })

    report = {
        "summary": summary,
        "insights": insights,
        "recommendations": recs,
        "ml_performance_score": round(ml_score, 1) if ml_score is not None else None,
    }
    return {"session_id": session_id, "report": report}


# -------------------------------------------
# 🧠 Standalone ML performance score prediction
# -------------------------------------------
@app.post("/api/predict/performance")
async def predict_performance(req: Request):
    """
    Lets the frontend (or a quick curl/Postman test) get an ML performance
    score for an arbitrary feature set without needing a saved session file.
    Body example:
      {"wpm": 145, "posture": 82, "fillers": 2, "volume": 1.1,
       "pitch": 180, "eyeContact": "Good", "tone": "Balanced", "emotion": "Confident"}
    """
    features = await req.json()
    score = predict_performance_score(features)
    if score is None:
        return {"error": "Performance model not loaded. Run backend/ml/train_performance_model.py first."}
    return {"ml_performance_score": round(score, 1)}


# -------------------------------------------
# 🧠 ML model health check
# -------------------------------------------
@app.get("/api/models/status")
async def model_status():
    """Quick way to confirm both ML models loaded successfully at startup."""
    return models_status()


# -------------------------------------------
# 📋 List all saved sessions (for Session History page)
# -------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    """
    Returns a lightweight summary of every saved session so the frontend
    can render a history table without fetching each full report.
    """
    sessions = []
    if os.path.isdir(RAW_DIR):
        for filename in os.listdir(RAW_DIR):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(RAW_DIR, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            sessions.append({
                "session_id": data.get("session_id", filename[:-5]),
                "timestamp": data.get("timestamp"),
                "wpm": data.get("avg_wpm", data.get("wpm", 0)),
                "posture": data.get("posture", 0),
                "emotion": data.get("emotion", "Neutral"),
            })

    # newest sessions first
    sessions.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    return {"sessions": sessions}


# -------------------------------------------
# 🏠 Root endpoint
# -------------------------------------------
@app.get("/")
async def root():
    return {"message": "Aura Coach backend running with real-time persistent speech analysis ✅"}
