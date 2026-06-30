import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import HomePage from "./pages/HomePage/HomePage";
import TrainingPage from "./pages/TrainingPage/TrainingPage";
import ReportPage from "./pages/ReportPage/ReportPage";
import SessionHistoryPage from "./pages/SessionHistoryPage/SessionHistoryPage";

// ✅ Load backend URL from environment variable
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8010";

console.log("🔗 Backend URL:", BACKEND_URL);

export default function App() {
  const [view, setView] = useState("home");
  const [sessionData, setSessionData] = useState({
    wpm: 0,
    posture: 0,
    eyeContact: "Unknown",
    fillers: 0,
    emotion: "Neutral",
  });
  const [sessionId, setSessionId] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // 🎥 Webcam setup — active only during training
  useEffect(() => {
    if (view === "training") {
      const startWebcam = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ 
            video: true, 
            audio: true 
          });
          if (videoRef.current) videoRef.current.srcObject = stream;
          streamRef.current = stream;
          console.log("✅ Webcam started");
        } catch (error) {
          console.error("❌ Webcam access error:", error);
          alert("Please allow camera and microphone access to continue.");
        }
      };
      startWebcam();
    } else {
      // Stop webcam when leaving training
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        console.log("🔴 Webcam stopped");
      }
    }
  }, [view]);

  // 🧩 Handle training session end — get real metrics
  const handleEndSession = async (metrics) => {
    try {
      console.log("📤 Sending session data:", metrics);
      
      const response = await fetch(`${BACKEND_URL}/api/session/log`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metrics),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("✅ Session logged successfully:", result);

      setSessionData(metrics);
      setSessionId(result.session_id);
      setView("report");
    } catch (error) {
      console.error("❌ Failed to log session:", error);
      alert(`Backend connection failed. Please check:\n1. FastAPI is running on port 8010\n2. CORS is properly configured\n3. Backend URL in .env is correct: ${BACKEND_URL}`);
    }
  };

  // 🔄 Navigation Handlers
  const handleStartSession = () => setView("training");
  const handleBackToHome = () => setView("home");
  const handleViewHistory = () => setView("history");

  return (
    <div className="App">
      {/* 🏠 Home Page */}
      {view === "home" && (
        <HomePage onStart={handleStartSession} onHistory={handleViewHistory} />
      )}

      {/* 🧘 Training Page */}
      {view === "training" && (
        <TrainingPage
          videoRef={videoRef}
          onEndSession={handleEndSession}
          onBack={handleBackToHome}
          streamRef={streamRef}
          backendUrl={BACKEND_URL}
        />
      )}

      {/* 📊 Report Page */}
      {view === "report" && (
        <ReportPage
          data={sessionData}
          sessionId={sessionId}
          onBack={handleBackToHome}
          backendUrl={BACKEND_URL}
        />
      )}

      {/* 🗂 Session History Page */}
      {view === "history" && (
        <SessionHistoryPage
          onBack={handleBackToHome}
          onOpenReport={(id) => {
            setSessionId(id);
            setView("report");
          }}
          backendUrl={BACKEND_URL}
        />
      )}
    </div>
  );
}
