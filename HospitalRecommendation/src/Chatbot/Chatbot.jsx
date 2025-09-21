import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./Chatbot.css";
import Header from "../Header/Header.jsx";

function Chatbot() {
  // recommendation basis: "specialization" | "city" | "auto"
  const [recBasis, setRecBasis] = useState("auto");
  const [selectedMethod, setSelectedMethod] = useState("cosine"); // cosine, knn, cluster
  const [showMetrics, setShowMetrics] = useState(false);
  const [metricsData, setMetricsData] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  const [messages, setMessages] = useState([
    { text: "Hello! Ask me about hospitals in Pakistan.", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

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
      setInput(speechText); // Fill input with speech-to-text result
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const sendMessage = async () => {
    if (input.trim()) {
      const feeRegex = /fee:\s*(\d\s*-\s*\d)/i;
      const feeMatch = input.match(feeRegex);
      const extractedFeeRange = feeMatch ? feeMatch[1] : "";
      const cleanedQuery = input.replace(feeRegex, "").trim();

      const userMessage = { text: input, sender: "user" };
      setMessages((prevMessages) => [...prevMessages, userMessage]);

      try {
        const response = await fetch("http://127.0.0.1:5000/chatbot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: cleanedQuery, feeRange: extractedFeeRange })
        });

        const data = await response.json();
        const hospitals = Array.isArray(data.response) ? data.response : [];

        // Save successful search into localStorage for recommendations
        if (hospitals.length > 0 && cleanedQuery) {
          const oldHistory = JSON.parse(localStorage.getItem("queryHistory") || "[]");
          const newHistory = [...oldHistory, cleanedQuery];
          localStorage.setItem("queryHistory", JSON.stringify(newHistory));
        }

        setMessages((prevMessages) => [
          ...prevMessages,
          hospitals.length > 0
            ? { hospitals, sender: "bot", kind: "search" }
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
    setMessages((prev) => [...prev, { text: "Recommendations", sender: "user" }]);

    try {
      const res = await fetch("http://127.0.0.1:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ history, basis: recBasis, method: selectedMethod })
      });
      const data = await res.json();
      const recs = data.recommendations || [];

      setMessages((prev) => [
        ...prev,
        recs.length > 0 ? { hospitals: recs, sender: "bot", kind: "recommendation" }
                        : { text: "No recommendations found.", sender: "bot" }
      ]);
    } catch (err) {
      console.error("fetchRecommendations error:", err);
      setMessages((prev) => [...prev, { text: "Error fetching recommendations.", sender: "bot" }]);
    }
  };

  // ✅ Fetch recommendation performance metrics
  const fetchPerformanceMetrics = async () => {
    const history = JSON.parse(localStorage.getItem("queryHistory") || "[]");
    
    if (history.length === 0) {
      setMessages((prev) => [...prev, { 
        text: "No search history available for performance metrics. Please search for some hospitals first.", 
        sender: "bot" 
      }]);
      return;
    }

    setLoadingMetrics(true);
    setMessages((prev) => [...prev, { text: "Performance Metrics", sender: "user" }]);
    
    try {
      const response = await fetch("http://127.0.0.1:5000/performance_metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          history: history,
          basis: recBasis,
          method: selectedMethod,
          top_k: 5
        })
      });

      const data = await response.json();
      
      if (data.status === "success") {
        setMetricsData(data.metrics);
        setShowMetrics(true);
        setMessages((prev) => [...prev, { 
          text: `Performance metrics calculated successfully! Based on ${history.length} search queries using ${selectedMethod} method.`, 
          sender: "bot" 
        }]);
      } else {
        setMessages((prev) => [...prev, { 
          text: `Error: ${data.error}`, 
          sender: "bot" 
        }]);
      }
    } catch (err) {
      console.error("Error fetching performance metrics:", err);
      setMessages((prev) => [...prev, { 
        text: "Error fetching performance metrics from server.", 
        sender: "bot" 
      }]);
    } finally {
      setLoadingMetrics(false);
    }
  };

  const formatValue = (value) => {
    if (typeof value === 'number') {
      return Number.isInteger(value) ? value : value.toFixed(4);
    }
    return value;
  };

  const getScoreClass = (value, type = "default") => {
    if (typeof value !== 'number') return '';
    
    switch (type) {
      case 'similarity':
      case 'relevance':
        if (value > 0.7) return 'score-excellent';
        if (value > 0.4) return 'score-good';
        return 'score-poor';
      case 'diversity':
        if (value > 0.6) return 'score-excellent';
        if (value > 0.3) return 'score-good';
        return 'score-poor';
      case 'coverage':
        if (value > 0.5) return 'score-excellent';
        if (value > 0.2) return 'score-good';
        return 'score-poor';
      default:
        return '';
    }
  };

  const renderNestedMetrics = (obj, prefix = '', level = 0) => {
    if (!obj || typeof obj !== 'object') return null;

    return Object.entries(obj).map(([key, value]) => {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        return (
          <div key={fullKey} className={`metric-section ${level > 0 ? 'nested' : ''}`}>
            <h4 className="metric-section-title">
              {key.replace(/_/g, ' ').toUpperCase()}
            </h4>
            <div className="metric-subsection">
              {renderNestedMetrics(value, fullKey, level + 1)}
            </div>
          </div>
        );
      }
      
      // Determine value type for color coding
      let valueType = 'default';
      if (key.includes('similarity') || key.includes('relevance')) valueType = 'similarity';
      if (key.includes('diversity')) valueType = 'diversity';
      if (key.includes('coverage') || key.includes('ratio')) valueType = 'coverage';
      
      return (
        <div key={fullKey} className="metric-item">
          <span className="metric-label">
            {key.replace(/_/g, ' ').toUpperCase()}:
          </span>
          <span className={`metric-value ${getScoreClass(value, valueType)}`}>
            {Array.isArray(value) ? `[${value.join(', ')}]` : formatValue(value)}
          </span>
        </div>
      );
    });
  };

  const renderMetrics = () => {
    if (!metricsData) return null;

    return (
      <div className="metrics-modal-overlay" onClick={() => setShowMetrics(false)}>
        <div className="metrics-modal" onClick={(e) => e.stopPropagation()}>
          <div className="metrics-header">
            <h2>Recommendation Performance Metrics</h2>
            <button 
              className="close-button" 
              onClick={() => setShowMetrics(false)}
            >
              ×
            </button>
          </div>
          <div className="metrics-content">
            {metricsData.error ? (
              <div className="error-message">
                {metricsData.error}
              </div>
            ) : (
              <>
                <div className="metrics-summary">
                  <h3>{metricsData.model_type}</h3>
                  <div className="summary-stats">
                    <div className="summary-item">
                      <span>Method:</span> <strong>{metricsData.method?.toUpperCase()}</strong>
                    </div>
                    <div className="summary-item">
                      <span>Basis:</span> <strong>{metricsData.basis?.toUpperCase()}</strong>
                    </div>
                    {metricsData.overall_metrics && (
                      <div className="summary-item overall-score">
                        <span>Overall Score:</span> 
                        <strong className={getScoreClass(metricsData.overall_metrics.overall_recommendation_score, 'similarity')}>
                          {formatValue(metricsData.overall_metrics.overall_recommendation_score)}
                        </strong>
                      </div>
                    )}
                  </div>
                  <p className="metrics-description">
                    📊 Performance analysis based on your search history and generated recommendations
                  </p>
                </div>
                
                <div className="metrics-grid">
                  {renderNestedMetrics(metricsData)}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
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
                  className={`hospital-list ${msg.kind === "recommendation" ? "recommend-box" : ""}`}
                >
                  {msg.hospitals.map((hospital, idx) => (
                    <div
                      key={idx}
                      className="hospital-card"
                      style={{
                        position: "relative",
                        ...(msg.kind === "recommendation"
                          ? { backgroundColor: "#e0f8e9", border: "2px solid #6A5ACD" }
                          : {})
                      }}
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
                          >
                            Visit
                          </a>
                        ) : (
                          "Not Available"
                        )}
                      </p>
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
        {/* Method and Basis selector */}
        <div className="controls-section">
          <div className="control-group">
            <label>Method:</label>
            <select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
              className="control-select"
            >
              <option value="cosine">TF-IDF Cosine</option>
              <option value="knn">KNN</option>
              <option value="cluster">K-Means Cluster</option>
            </select>
          </div>
          
          <div className="control-group">
            <label>Basis:</label>
            <select
              value={recBasis}
              onChange={(e) => setRecBasis(e.target.value)}
              className="control-select"
            >
              <option value="auto">Auto (Combined)</option>
              <option value="specialization">Specialization</option>
              <option value="city">City</option>
            </select>
          </div>
        </div>

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
          id="recommendButton"
          onClick={fetchRecommendations}
          className="action-button recommend-btn"
        >
          Get Recommendations
        </button>

        {/* ✅ Performance Metrics Button */}
        <button
          id="metricsButton"
          onClick={fetchPerformanceMetrics}
          disabled={loadingMetrics}
          className="action-button metrics-btn"
        >
          {loadingMetrics ? "Loading..." : "Performance Metrics"}
        </button>
      </div>

      {/* ✅ Render metrics modal */}
      {showMetrics && renderMetrics()}
    </>
  );
}

export default Chatbot;