import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [metricsData, setMetricsData] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState('FedAvg'); // Default strategy
  const [selectedModel, setSelectedModel] = useState('CNN'); // Default strategy
  const [logOpen, setLogOpen] = useState(false);

  // Function to fetch metrics and update state
  const fetchMetrics = async () => {
    try {
      console.log('Hi');
      const response = await fetch('http://localhost:80/metrics');
      console.log(response);
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
    }, 500); // Fetch every 5 seconds (adjust as needed)

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures that this effect runs once

  const handleStrategyChange = (event) => {
    setSelectedStrategy(event.target.value);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const toggleLog = () => {
    setLogOpen(!logOpen);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Federated Learning Experiment Dashboard</h1>
      </header>
      <hr/>

      <div className='Inputs'>
        <div className='strategy'>
          <h3>Select Strategy Type</h3>
          <select value={selectedStrategy} onChange={handleStrategyChange}>
            <option value="FedAvg">FedAvg</option>
            <option value="FedProx">FedProx</option>
            <option value="FedAdam">FedAdam</option>
            <option value="Krum">Krum</option>
            <option value="Bulyan">Bulyan</option>
          </select>
        </div>

        <div className='parameters'>
          <h3>Parameters</h3>
          <div className="parameter-inputs">
            <div className="parameter-input">
              <input type="text" id="learning-rate" name="learning-rate" placeholder='Learning rate'/>
            </div>
            <div className="parameter-input">
              <input type="text" id="momentum" name="momentum" placeholder='Momentum'/>
            </div>
          </div>
        </div>

        <div className='model'>
          <h3>Model Selection</h3>
          <select value={selectedModel} onChange={handleModelChange}>
            <option value="FedAvg">CNN</option>
            <option value="FedProx">ResNet</option>
            <option value="FedAdam">BERT</option>
            <option value="Krum">GPT</option>
            <option value="Bulyan">LSTM</option>
          </select>
        </div>

        <div className='rounds'>
          <h3>Number of Rounds</h3>
          <div className="round-input">
              <input type="text" id="round" name="rounds" placeholder='Rounds'/>
          </div>
        </div>
      </div>
      <hr/>

      <div className='start'>
        <button>Start Experiment</button>
      </div>
      <hr/>

      <div className='topology'>
        <h2> Topology Visualization </h2>
        <h3> (TBD) </h3>
      </div>
      <hr/>

      <div className='output'>
        {/* Button to toggle the experiment log */}
        <button onClick={toggleLog}>{logOpen ? 'Close Experiment Log' : 'Open Experiment Log'}</button>

        {/* Experiment log */}
        {logOpen && (
          <div className="experiment-log">
            <h2>Output Metrics</h2>
            {/* Display metrics data */}
            {metricsData && (
              <div>
                {metricsData.map((item, index) => (
                  <div key={index}>
                    <p>Round {index + 1} accuracy, loss: {item.accuracy}, {item.loss}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
