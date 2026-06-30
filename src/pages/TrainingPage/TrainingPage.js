import React, { useRef, useEffect, useState } from "react";
import * as faceapi from "face-api.js";
import { Pose } from "@mediapipe/pose";
import { Camera } from "@mediapipe/camera_utils";
import "./TrainingPage.css";
import bg from "../../assets/background.png";

export default function TrainingPage({ onEndSession, onBack, streamRef, backendUrl }) {
  const videoRef = useRef(null);
  const recognitionRef = useRef(null);
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const audioStreamRef = useRef(null);

  // Use the same backend URL logic as the rest of the app instead of hardcoding it
  const BACKEND_URL = backendUrl || process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8010";
  // Convert http(s) backend URL into the matching ws(s) URL for the audio stream
  const WS_URL = BACKEND_URL.replace(/^http/, "ws") + "/api/audio-stream";

  const [metrics, setMetrics] = useState({
    posture: 0,
    headTilt: 0,
    eyeContact: "Unknown",
    emotion: "Detecting...",
    wpm: 0,
    fillers: 0,
    volume: 0,
    pitch: 0,
    tone: "Neutral",
  });

  const [error, setError] = useState(null);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  // Pose + Face detection
  useEffect(() => {
    const videoEl = videoRef.current;
    let camera = null;
    let emotionInterval = null;

    const pose = new Pose({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

   pose.onResults((res) => {
  if (res.poseLandmarks) {
    const lm = res.poseLandmarks;
    const L = lm[11], R = lm[12], nose = lm[0];

    if (L && R && nose) {
      const shoulderTilt = (R.y - L.y) * 100;
      const postureScore = Math.max(0, 100 - Math.abs(shoulderTilt) * 4);

      //  FIXED: normalize head tilt so looking straight ≈ 0°
      let rawTilt = Math.atan2(R.y - L.y, R.x - L.x) * (180 / Math.PI);

      // Converts -180~180 into a small tilt angle around 0
      if (rawTilt > 90) rawTilt -= 180;
      if (rawTilt < -90) rawTilt += 180;

      const headTilt = Number(rawTilt.toFixed(0));

      const eyeContact =
        Math.abs(nose.x - 0.5) < 0.05 ? "Good" : "Looking Away";

      setMetrics((prev) => ({
        ...prev,
        posture: Number(postureScore.toFixed(0)),
        headTilt,
        eyeContact,
      }));
    }
  }
});


    const startCamera = async () => {
      try {
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
        await faceapi.nets.faceExpressionNet.loadFromUri("/models");

        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoEl) {
          videoEl.srcObject = stream;
          if (streamRef) streamRef.current = stream;
        }

        camera = new Camera(videoEl, {
          onFrame: async () => { if (isSessionActive) await pose.send({ image: videoEl }); },
          width: 1280,
          height: 720,
        });
        camera.start();

        emotionInterval = setInterval(async () => {
          if (!isSessionActive || !videoEl || videoEl.readyState < 2) return;
          const detections = await faceapi.detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions();
          if (detections && detections.expressions) {
            const sorted = Object.entries(detections.expressions).sort((a,b)=>b[1]-a[1]);
            const emotion = sorted[0][0];
            setMetrics(prev => ({ ...prev, emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1) }));
          }
        }, 1200);
      } catch (e) {
        console.error(e);
        setError("Unable to access camera or load models.");
      }
    };

    startCamera();

    return () => {
      if (camera) camera.stop();
      if (emotionInterval) clearInterval(emotionInterval);
      if (streamRef?.current) {
        streamRef.current.getTracks().forEach(t=>t.stop());
        streamRef.current = null;
      }
    };
  }, [isSessionActive, streamRef]);

  // Audio WebSocket stream
  const startAudioStreaming = async (sid = null) => {
  try {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("🟢 WebSocket audio connected");
      if (sid) ws.send(JSON.stringify({ session_id: sid }));
    };

    ws.onerror = (err) => console.error("WS error:", err);
    ws.onclose = () => console.log("🔴 WS closed");

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioStreamRef.current = stream;

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = audioCtx;

    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      if (ws.readyState === WebSocket.OPEN) {
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) pcm16[i] = input[i] * 0x7fff;
        ws.send(pcm16.buffer);
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

  } catch (err) {
    console.error("Audio error:", err);
    setError("Microphone access failed.");
  }
};


  const stopAudioStreaming = () => {
    try {
      wsRef.current?.close();
      audioStreamRef.current?.getTracks().forEach((t) => t.stop());
      processorRef.current?.disconnect();
      audioCtxRef.current?.close();
    } catch (e) {
      console.warn("stopAudioStreaming cleanup", e);
    } finally {
      wsRef.current = null;
      audioStreamRef.current = null;
      processorRef.current = null;
      audioCtxRef.current = null;
    }
  };

  // Speech recognition (browser)
  const startSpeechRecognition = () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    let wordCount = 0;
    let fillerCount = 0;
    const fillers = ["um", "uh", "like", "basically", "you know"];
    const startTime = Date.now();

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        transcript += event.results[i][0].transcript.toLowerCase();
      }
      const words = transcript.split(/\s+/).filter(Boolean);
      wordCount += words.length;
      fillerCount += words.filter((w) => fillers.includes(w)).length;
      const elapsedMinutes = (Date.now() - startTime) / 60000;
      const wpm = elapsedMinutes > 0 ? Math.round(wordCount / elapsedMinutes) : 0;

      // small frontend guard against music false positives
      if (words.length < 3 && wpm < 40) return;

      setMetrics(prev => ({ ...prev, wpm, fillers: fillerCount }));
    };

    recognition.onerror = (event) => console.error("Speech error:", event.error);
    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
  };

  // Start/end session
  const handleSessionToggle = async () => {
    if (isSessionActive) {
      stopSpeechRecognition();
      stopAudioStreaming();
      setIsSessionActive(false);

      // send final metrics to backend and get session_id
      try {
        const response = await fetch(`${BACKEND_URL}/api/session/log`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(metrics),
        });
        const result = await response.json();
        setSessionId(result.session_id);
        // go to report (parent handles view; s implicit with onEndSession)
        onEndSession({ ...metrics, session_id: result.session_id });
      } catch (err) {
        console.error("Error saving session:", err);
        onEndSession(metrics);
      }
    } else {
      // create a session id first so backend can link live audio chunks
      const tempSessionId = `session_${Math.random().toString(36).slice(2,9)}`;
      setSessionId(tempSessionId);
      startSpeechRecognition();
      startAudioStreaming(tempSessionId); // send session id on WS open
      setIsSessionActive(true);
    }
  };

  return (
    <div className="training-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
        <button className="home-link" onClick={onBack}>← Home</button>
      </header>

      <div className="grid">
        {/* Left */}
        <section className="card glass card-left">
          <h2 className="card-title">Body Language Analysis</h2>
          <div className="stat"><div className="stat-label">Posture Score</div><div className="stat-value accent">{metrics.posture}%</div></div>
          <div className="stat"><div className="stat-label">Head Tilt</div><div className="stat-value">{metrics.headTilt}°</div></div>
          <div className="stat"><div className="stat-label">Eye Contact</div><div className="stat-value">{metrics.eyeContact}</div></div>
        </section>

        {/* Center */}
        <section className="card glass card-center">
          <div className="video-frame">
            {error ? <div className="error">{error}</div> : <video ref={videoRef} autoPlay playsInline muted className="video" />}
          </div>

          {/* Live tips */}
          <div className="live-tips-card">
  <h3>🔮 Live Tips</h3>

  <div className="tips-list">
    {metrics.posture < 60 && <p> Straighten your posture</p>}
    {metrics.eyeContact !== "Good" && <p> Maintain eye contact</p>}
    {metrics.volume < 10 && <p> Speak louder</p>}
    {metrics.fillers > 5 && <p> Reduce filler words</p>}
    {metrics.tone === "Energetic" && <p> Great energy!</p>}

    {(metrics.posture >= 60 &&
      metrics.eyeContact === "Good" &&
      metrics.volume >= 10 &&
      metrics.fillers <= 5 &&
      metrics.tone !== "Energetic") && (
      <p>✅ You’re doing great!</p>
    )}
  </div>
</div>


          <div className="actions">
            <button className={`btn ${isSessionActive ? "danger" : "start"}`} onClick={handleSessionToggle}>
              {isSessionActive ? "⏹ End Session & View Report" : "▶ Start Session"}
            </button>
          </div>
        </section>

        {/* Right */}
        <section className="card glass card-right">
          <h2 className="card-title">Speaking Style Feedback</h2>
          <div className="stat"><div className="stat-label">Speaking Pace</div><div className="stat-value accent">{metrics.wpm} WPM</div></div>
          <div className="stat"><div className="stat-label">Filler Words</div><div className="stat-value">{metrics.fillers}</div></div>
          <div className="stat"><div className="stat-label">Emotion</div><div className="stat-value">{metrics.emotion}</div></div>
          <div className="stat"><div className="stat-label">Tone</div><div className="stat-value">{metrics.tone}</div></div>
          <div className="stat"><div className="stat-label">Volume</div><div className="stat-value">{(metrics.volume||0).toFixed(1)}</div></div>
          <div className="stat"><div className="stat-label">Pitch</div><div className="stat-value">{metrics.pitch}</div></div>
        </section>
      </div>
    </div>
  );
}
