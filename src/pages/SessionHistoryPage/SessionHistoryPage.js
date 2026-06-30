import React, { useEffect, useState } from "react";
import "./SessionHistoryPage.css";
import bg from "../../assets/background.png";

export default function SessionHistoryPage({ onBack, onOpenReport, backendUrl }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  const BACKEND_URL = backendUrl || process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8010";

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/sessions`);
        const data = await res.json();
        setSessions(data.sessions || []);
      } catch (err) {
        console.error("Error fetching sessions:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchSessions();
  }, [BACKEND_URL]);

  if (loading) return <div className="loading glass">Loading sessions...</div>;

  return (
    <div className="history-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
        <button className="home-link" onClick={onBack}>
          ← Back
        </button>
      </header>

      <div className="history-container glass">
        <h2>Previous Sessions</h2>

        {sessions.length === 0 ? (
          <p>No saved sessions yet.</p>
        ) : (
          <table className="session-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>WPM</th>
                <th>Posture</th>
                <th>Emotion</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id}>
                  <td>{new Date(s.timestamp).toLocaleString()}</td>
                  <td>{s.wpm}</td>
                  <td>{s.posture}%</td>
                  <td>{s.emotion}</td>
                  <td>
                    <button
                      className="btn small"
                      onClick={() => onOpenReport(s.session_id)}
                    >
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
