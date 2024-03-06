import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [metricsData, setMetricsData] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState('FedAvg'); // Default strategy
  const [selectedDataset, setSelectedDataset] = useState('CIFAR-10'); // Default dataset
  const [selectedModel, setSelectedModel] = useState('CNN'); // Default model
  const [logOpen, setLogOpen] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [isNonIID, setIsNonIID] = useState(false); // Track if non-IID data is selected
  const [numGroups, setNumGroups] = useState(''); // Number of groups input value

  // Function to fetch metrics and update state
  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:80/metrics'); // Locally handling CORS error
      const data = await response.json();
      setMetricsData(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  // Set up periodic fetching using useEffect and setInterval
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchMetrics();
    }, 8000); // Fetch every 8 seconds (adjust as needed)

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array ensures that this effect runs once

  const handleStrategyChange = (event) => {
    setSelectedStrategy(event.target.value);
  };

  const handleDatasetChange = (event) => {
    setSelectedDataset(event.target.value);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const toggleLog = () => {
    setLogOpen(!logOpen);
  };

  function handleMetricsChange(event) {
    const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
    console.log(selectedOptions);
    setSelectedMetrics(selectedOptions);
  }

  // Function to handle radio button change for IID and non-IID data
  const handleDataTypeChange = (event) => {
    if (event.target.value === 'non-iid') {
      setIsNonIID(true);
    } else {
      setIsNonIID(false);
      setNumGroups(''); // Reset number of groups input value
    }
  };

  // TO DO: file input for model select
  const handleFileChange = (event) => {
    const file = event.target.files[0];
  }

  // Function to handle input change for number of groups
  const handleNumGroupsChange = (event) => {
    setNumGroups(event.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Federated Learning Experiment Dashboard</h1>
      </header>
      <hr/>

      <div className='Inputs'>
        <div className='strategy'>
          <h3>Strategy Type</h3>
          <select value={selectedStrategy} onChange={handleStrategyChange}>
            <option value="FedAvg">FedAvg</option>
            <option value="FedProx">FedProx</option>
            <option value="FedAdam">FedAdam</option>
            <option value="Krum">Krum</option>
            <option value="Bulyan">Bulyan</option>
          </select>
        </div>

        <div className='dataset'>
          <h3>Dataset</h3>
          <select value={selectedDataset} onChange={handleDatasetChange}>
            <option value="CIFAR-10">CIFAR-10</option>
          </select>
        </div>

        {/*TO DO: Align model preselect and file input*/}
        <div className='model'>
          <h3>Model Selection</h3>
          <div>
            {/* UI Error: <text style={{marginRight: '10px'}}>Preselect: </text> */}
            <select value={selectedModel} onChange={handleModelChange}>
              <option value="FedAvg">CNN</option>
              <option value="FedProx">ResNet</option>
              <option value="FedAdam">BERT</option>
              <option value="Krum">GPT</option>
              <option value="Bulyan">LSTM</option>
            </select>
          </div>
          <div>
            {/* UI Error: <text style={{marginRight: '10px'}}>File: </text> */}
            <input type="file" onChange={handleFileChange} />
          </div>
        </div>

        <div className='parameters'>
          <h3>Hyperparameters</h3>
          <div className="parameter-inputs">
            <div className="parameter-input">
              <input type="text" id="learning-rate" name="learning-rate" placeholder='Learning rate'/>
            </div>
            <div className="parameter-input">
              <input type="text" id="momentum" name="momentum" placeholder='Momentum'/>
            </div>
          </div>
        </div>

        {/*<div className='rounds'>
          <h3>Number of Rounds</h3>
          <div className="round-input">
              <input type="text" id="round" name="rounds" placeholder='Rounds'/>
          </div>
        </div>*/}

        <div className='distribution'>
          <h3>Data Distribution</h3>
          <div>
            <input type="radio" id="iid" name="data-type" value="iid" checked={!isNonIID} onChange={handleDataTypeChange}/>
            <label htmlFor="iid">IID Data</label>
          </div>

          <div>
            <input type="radio" id="non-iid" name="data-type" value="non-iid" checked={isNonIID} onChange={handleDataTypeChange}/>
            <label htmlFor="non-iid">Non-IID Data</label>
          </div>

          <div>
            <input type="number" id="num-groups" placeholder="Number of groups" disabled={!isNonIID} value={numGroups} 
            onChange={handleNumGroupsChange}/>
          </div>
        </div>

        <div className='metrics'>
          <h3>Output Metrics</h3>
          <select value={selectedMetrics} onChange={handleMetricsChange} multiple>
            <option value="Accuracy">Accuracy</option>
            <option value="Loss">Loss</option>
            <option value="Convergence Speed">Convergence Speed</option>
            <option value="Delay">Delay</option>
          </select>
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
