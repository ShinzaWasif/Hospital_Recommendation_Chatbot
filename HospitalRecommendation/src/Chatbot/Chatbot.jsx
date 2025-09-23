import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./Chatbot.css";
import Header from "../Header/Header.jsx";

function Chatbot() {
  const [messages, setMessages] = useState([
    { text: "Hello! Ask me about hospitals in Pakistan.", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const [metrics, setMetrics] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);

  // Function to handle speech input
  const startListening = () => {
    if (!("webkitSpeechRecognition" in window)) {
      alert("Speech recognition is not supported in your browser.");
      return;
    }

    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const speechText = event.results[0][0].transcript;
      setInput(speechText);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const sendMessage = async () => {
    if (input.trim()) {
      const userMessage = { text: input, sender: "user" };
      setMessages((prevMessages) => [...prevMessages, userMessage]);

      try {
        const response = await fetch("http://127.0.0.1:5000/chatbot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: input.trim() })
        });

        const data = await response.json();
        const hospitals = Array.isArray(data.response) ? data.response : [];

        // Save successful search into localStorage for recommendations
        if (hospitals.length > 0 && input.trim()) {
          const oldHistory = JSON.parse(localStorage.getItem("queryHistory") || "[]");
          const newHistory = [...oldHistory, input.trim()];
          localStorage.setItem("queryHistory", JSON.stringify(newHistory));
        }

        // ✅ Simple search mein recommendation style nahi use karo
        setMessages((prevMessages) => [
          ...prevMessages,
          hospitals.length > 0
            ? { hospitals, sender: "bot" } // ❌ 'kind: "recommendation"' remove karo
            : { text: "Sorry, no matching hospitals found.", sender: "bot" }
        ]);
      } catch (error) {
        console.error("sendMessage error:", error);
        setMessages((prevMessages) => [...prevMessages, { text: "Error fetching response.", sender: "bot" }]);
      }
      setInput("");
    }
  };

  const fetchRecommendations = async () => {
    const history = JSON.parse(localStorage.getItem("queryHistory") || "[]");

    if (history.length === 0) {
      setMessages((prev) => [...prev, { text: "No search history yet for recommendations.", sender: "bot" }]);
      return;
    }

    // ✅ Pehle "Getting recommendations..." message show karo
    setMessages((prev) => [...prev, { text: "Getting recommendations based on your search history...", sender: "bot" }]);

    try {
      const res = await fetch("http://127.0.0.1:5000/recommend_knn_city_spec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          history: history, 
          top_k: 5 
        })
      });
      
      const data = await res.json();
      const recs = data.recommendations || [];
      
      // ✅ Recommendations ke liye special style use karo
      setMessages((prev) => [
        ...prev,
        recs.length > 0 ? { 
          hospitals: recs, 
          sender: "bot", 
          kind: "recommendation" // ✅ Sirf recommendations ke liye
        } : { 
          text: "No recommendations found.", 
          sender: "bot" 
        }
      ]);
    } catch (err) {
      console.error("fetchRecommendations error:", err);
      setMessages((prev) => [...prev, { text: "Error fetching recommendations.", sender: "bot" }]);
    }
  };

  // ✅ fetch metrics from backend
  const fetchMetrics = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/metrics");
      const data = await res.json();
      setMetrics(data);
      setShowMetrics(true);
    } catch (err) {
      console.error("Error fetching metrics:", err);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <>
      <Header />
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => {
            if (msg.sender === "bot" && msg.hospitals) {
              return (
                <div
                  key={index}
                  className={`hospital-list ${msg.kind === "recommendation" ? "recommendation-section" : "search-results"}`}
                >
                  {/* ✅ Recommendation heading agar recommendation hai to */}
                  {msg.kind === "recommendation" && (
                    <div className="recommendation-header">
                      {/* <h3>💫 Recommended Hospitals Based on Your Search History</h3> */}
                    </div>
                  )}
                  
                  {msg.hospitals.map((hospital, idx) => (
                    <div
                      key={idx}
                      className={`hospital-card ${msg.kind === "recommendation" ? "recommendation-card" : "search-card"}`}
                    >
                      <h3>{hospital["Name"]}</h3>
                      <p><strong>City:</strong> {hospital["City"]}</p>
                      <p><strong>Province:</strong> {hospital["Province"]}</p>
                      <p><strong>Specialization:</strong> {hospital["Specialization"]}</p>
                      <p><strong>Phone:</strong> {hospital["Phone"]}</p>
                      <p><strong>Contact Person:</strong> {hospital["ContactPerson"]}</p>
                      <p><strong>Address:</strong> {hospital["Address"]}</p>
                      <p><strong>Fees:</strong> {hospital["Fees"]}</p>
                      <p>
                        <strong>Website:</strong>{" "}
                        {hospital["Website"] ? (
                          <a
                            href={hospital["Website"]}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="website-link"
                          >
                            Visit
                          </a>
                        ) : (
                          "Not Available"
                        )}
                      </p>
                      
                      {/* ✅ Recommendation badge sirf recommendations ke liye */}
                      {msg.kind === "recommendation" && (
                        <div className="recommendation-badge">
                          ⭐ Recommended
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              );
            } else {
              return (
                <div
                  key={index}
                  className={msg.sender === "bot" ? "bot-message" : "user-message"}
                >
                  {msg.text}
                </div>
              );
            }
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input controls */}
      <div id="chatInputContainer">
        <input
          id="chatInput"
          type="text"
          placeholder='Ask about hospitals... (e.g., "cancer treatment hospital in lahore fee 30000-60000")'
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button id="micButton" onClick={startListening}>
          <i className="fa fa-microphone text-xl"></i>
        </button>
        <button id="sendButton" onClick={sendMessage}>
          <i className="fa fa-paper-plane text-xl"></i>
        </button>

        <button
          onClick={fetchRecommendations}
          style={{ 
            marginLeft: "8px", 
            padding: "8px 12px", 
            borderRadius: 10, 
            border: "none", 
            cursor: "pointer", 
            background: "#6A5ACD", 
            color: "white",
            fontWeight: "bold"
          }}
        >
          💫 Get Recommendations
        </button>

        <button
          onClick={fetchMetrics}
          style={{ 
            marginLeft: "8px", 
            padding: "8px 12px", 
            borderRadius: 10, 
            border: "none", 
            cursor: "pointer", 
            background: "#FF6B6B", 
            color: "white",
            fontWeight: "bold"
          }}
        >
          📊 Metrics
        </button>
      </div>

      {/* Metrics Display Modal */}
      {showMetrics && metrics && (
        <div className="metrics-modal">
          <h3>System Performance Metrics</h3>
          <div className="metric-item">
            <strong>Precision:</strong> <span className="metric-value">{metrics.precision}</span>
          </div>
          <div className="metric-item">
            <strong>Recall:</strong> <span className="metric-value">{metrics.recall}</span>
          </div>
          <div className="metric-item">
            <strong>F1 Score:</strong> <span className="metric-value">{metrics.f1}</span>
          </div>
          <button 
            onClick={() => setShowMetrics(false)}
            className="close-metrics-btn"
          >
            Close
          </button>
        </div>
      )}
    </>
  );
}

export default Chatbot;