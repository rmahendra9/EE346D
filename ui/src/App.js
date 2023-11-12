import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [metricsData, setMetricsData] = useState(null);

  // Function to fetch metrics and update state
  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:5000/metrics');
      const data = await response.json();
      console.log(data);
      setMetricsData(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  // Set up periodic fetching using useEffect and setInterval
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchMetrics();
    }, 5000); // Fetch every 5 seconds (adjust as needed)

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures that this effect runs once

  return (
    <div className="App">
      <header className="App-header">
        <h2>Weighted Average Accuracy</h2>

        {/* Display metrics data */}
        {metricsData && (
          <div>
            {metricsData.map((item, index) => (
              <div key={index}>
                <p>Round {index + 1} accuracy: {item.accuracy}</p>
              </div>
            ))}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
