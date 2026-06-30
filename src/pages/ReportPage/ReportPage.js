import React, { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";
import "./ReportPage.css";
import bg from "../../assets/background.png";

// Accept backendUrl as an optional prop (recommended for flexibility)
export default function ReportPage({ data, sessionId, onBack, backendUrl }) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  // Use the dynamic backend URL logic below
  const BACKEND_URL =
    backendUrl ||
    process.env.REACT_APP_BACKEND_URL ||
    "http://127.0.0.1:8010";

  // 🧠 Fetch report from FastAPI
  useEffect(() => {
    const fetchReport = async () => {
      if (!sessionId) return;
      try {
        const response = await fetch(
          `${BACKEND_URL}/api/report/${sessionId}`
        );
        const result = await response.json();
        console.log("📊 Report fetched:", result);
        setReport(result.report);
      } catch (error) {
        console.error("❌ Error fetching report:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
    // eslint-disable-next-line
  }, [sessionId, BACKEND_URL]);

  // 📈 Draw chart AFTER report loads
  useEffect(() => {
    if (!report || !data || !chartRef.current) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const ctx = chartRef.current.getContext("2d");

    // Dynamic simulated trend data
    const postureTrend = [
      Math.max(50, data.posture - 10),
      Math.max(55, data.posture - 5),
      data.posture,
      Math.min(95, data.posture + 5),
    ];
    const wpmTrend = [
      Math.max(80, data.wpm - 30),
      data.wpm - 10,
      data.wpm,
      Math.min(180, data.wpm + 15),
    ];

    chartInstanceRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: ["Start", "1 min", "2 min", "3 min", "End"],
        datasets: [
          {
            label: "Confidence (Posture)",
            data: postureTrend,
            borderColor: "rgba(125, 82, 245, 0.9)",
            backgroundColor: "rgba(110, 92, 226, 0.25)",
            tension: 0.4,
            fill: true,
            borderWidth: 2.5,
          },
          {
            label: "Pacing (WPM)",
            data: wpmTrend,
            borderColor: "rgba(34,197,94,0.9)",
            backgroundColor: "rgba(74,222,128,0.2)",
            tension: 0.4,
            fill: true,
            borderWidth: 2.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1200, easing: "easeInOutQuart" },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 100,
            ticks: { color: "#4B5563", font: { size: 13 } },
          },
          x: {
            ticks: { color: "#6B7280", font: { size: 12 } },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#4B5563", font: { size: 13, weight: 600 } },
          },
        },
      },
    });
    // eslint-disable-next-line
  }, [report, data]);

  if (loading)
    return <p className="loading glass">Loading report...</p>;

  if (!report)
    return (
      <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
        <header className="topbar glass">
          <div className="brand">Aura Coach</div>
        </header>
        <div className="report-container glass">
          <h2>No report found 😕</h2>
          <button className="btn start" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </div>
    );

  return (
    <div className="report-root" style={{ backgroundImage: `url(${bg})` }}>
      <header className="topbar glass">
        <div className="brand">Aura Coach</div>
      </header>

      <div className="report-container">
        {/* Summary */}
        <div className="glass overview-box">
          <h2>Session Summary</h2>
          <p>{report.summary}</p>
          {report.ml_performance_score !== null && report.ml_performance_score !== undefined && (
            <div className="ml-score-badge">
              <span className="ml-score-label">🧠 ML Performance Score</span>
              <span className="ml-score-value">{report.ml_performance_score}/100</span>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="stats-row">
          <div className="glass box">
            <h3>WPM</h3>
            <p className="big">{data.wpm}</p>
            <span>Words per Minute</span>
          </div>
          <div className="glass box">
            <h3>Posture</h3>
            <p className="big">{data.posture}%</p>
            <span>Body Alignment</span>
          </div>
          <div className="glass box">
            <h3>Eye Contact</h3>
            <p className="big">{data.eyeContact}</p>
            <span>Engagement</span>
          </div>
          <div className="glass box">
            <h3>Volume</h3>
            <p className="big">{data.volume?.toFixed(1)}</p>
            <span>Voice Power</span>
          </div>
          <div className="glass box">
            <h3>Emotion</h3>
            <p className="big">{data.emotion}</p>
            <span>Expression</span>
          </div>
        </div>

        {/* Chart + Insights */}
        <div className="middle-row">
          <div className="glass chart-box" style={{ height: "320px" }}>
            <h2>Pacing & Confidence Trend</h2>
            <canvas ref={chartRef}></canvas>
          </div>

          <div className="glass feedback-box">
            <h2>Insights</h2>
            <ul>
              <li>{report.insights.posture}</li>
              <li>{report.insights.eye_contact}</li>
              <li>{report.insights.emotion}</li>
              {report.insights.tone && <li>{report.insights.tone}</li>}
            </ul>
          </div>
        </div>

        {/* Recommendations */}
        <div className="glass feedback-box">
          <h2>Recommendations</h2>
          <ul>
            {report.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>

        <div className="actions">
          <button className="btn start" onClick={onBack}>
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
