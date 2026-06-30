import React, { useEffect, useState } from "react";
import "./HomePage.css";
import bg from "../../assets/background.png";

export default function HomePage({ onStart, onHistory }) {
  const fullText = "Face. Train. Succeed.";
  const [displayText, setDisplayText] = useState("");

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setDisplayText(fullText.slice(0, i + 1));
      i++;
      if (i === fullText.length) {
        setTimeout(() => {
          i = 0;
          setDisplayText("");
        }, 2000);
      }
    }, 150);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="home-container" style={{ backgroundImage: `url(${bg})` }}>
      {/* 🟣 Top-right user bar */}
      <div className="user-bar fade-in">
        <div className="user-circle glow"></div>
        <div className="user-name">NAME</div>
      </div>

      {/* 🟣 Center content */}
      <div className="center-content">

  {/* ✅ Text is now positioned absolutely so nothing moves */}
  <div className="hero-title-wrapper">
    <h1 className="hero-title typing-text">{displayText}</h1>
  </div>
  <div className="button-wrapper">
    <button className="start-btn" onClick={onStart}>
      ▶ Start Training
      </button>
    <button className="start-btn" onClick={onHistory} style={{ marginTop: "12px" }}>
      🗂 Session History
      </button>
      </div>
    </div>
    </div>
  );
}