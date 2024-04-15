import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import "chart.js/auto";
import './App.css';
import TopologyVisualization from './TopologyVisualization';

function App() {
  const [selectedStrategy, setSelectedStrategy] = useState('FedAvg'); // Default strategy
  const [selectedDataset, setSelectedDataset] = useState('CIFAR-10'); // Default dataset
  const [selectedModel, setSelectedModel] = useState('CNN'); // Default model
  const [learningRate, setLearningRate] = useState(0);
  const [linkDelay, setLinkDelay] = useState(0);
  const [momentum, setMomentum] = useState(0);
  const [rounds, setRounds] = useState(0);
  const [newAccuracyData, setAccuracyData] = useState({
    labels: [],
    datasets: [{
      label: 'CNN',
      data: [],
      color: 'blue',
      backgroundColor: 'blue'
    }]
  });
  const [newLossData, setLossData] = useState({
    labels: [],
    datasets: [{
      label: 'CNN',
      data: [],
      color: 'red',
      backgroundColor: 'red'
    }]
  });
  const lossOptions = {
    scales: {
      x: {
        type: 'category', // Set the x-axis type to 'category' for labels
        title: {
          display: true,
          text: 'Step' // Set the x-axis label text
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Loss', // Set the chart title
        font: {
          size: 15 // Adjust the font size of the title
        }
      }
    }
  };
  const accOptions = {
    scales: {
      x: {
        type: 'category', // Set the x-axis type to 'category' for labels
        title: {
          display: true,
          text: 'Step' // Set the x-axis label text
        }
      }
    },
    plugins: {
      title: {
        display: true,
        text: 'Accuracy', // Set the chart title
        font: {
          size: 15 // Adjust the font size of the title
        }
      }
    }
  };

  const adjacencyList = {
    A: ['B', 'C'],
    B: ['C', 'D'],
    C: ['D', 'E'],
    D: ['E'],
    E: []
  };

  // Function to fetch metrics and update state
  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:80/metrics'); // Locally handling CORS error
      const data = await response.json();
      if (data) {
        const newAccuracyData = {
          labels: [],
          datasets: [{
            label: 'CNN',
            data: [],
            color: 'blue',
            backgroundColor: 'blue'
          }]
        };
        const newLossData = {
          labels: [],
          datasets: [{
            label: 'CNN',
            data: [],
            color: 'red',
            backgroundColor: 'red'
          }]
        };
        data.forEach((item, index) => {
          newAccuracyData.labels.push(index + 1);
          newAccuracyData.datasets[0].data.push(item.accuracy);
          newLossData.labels.push(index + 1);
          newLossData.datasets[0].data.push(item.loss);
        });
        setAccuracyData(newAccuracyData);
        setLossData(newLossData);
      }
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  // Set up periodic fetching using useEffect and setInterval
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchMetrics();
    }, 800); // Fetch every 0.8 seconds (adjust as needed)

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


  const handleLearningRateChange = (event) => {
    setLearningRate(event.target.value);
  };

  const handleMomentumChange = (event) => {
    setMomentum(event.target.value);
  };

  const handleRoundsChange = (event) => {
    setRounds(event.target.value);
  };

  const handleLinkDelayChange = (event) => {
    setLinkDelay(event.target.value);
  };

  const handleStartExperiment = async () => {
    try {
      // console.log(selectedStrategy);
      // console.log(selectedModel);
      // console.log(selectedDataset);
      // console.log(learningRate);
      // console.log(linkDelay);
      // console.log(momentum);
      // console.log(rounds);
      const response = await fetch('http://localhost:80/start-experiment', {
        method: 'POST',
        body: JSON.stringify({
          strategy: selectedStrategy,
          model: selectedModel,
          learningRate: learningRate,
          momentum: momentum,
          rounds: rounds,
          linkDelay: linkDelay
        }),
        headers: {
          "Content-type": "application/json; charset=UTF-8"
        }
      });
      if (response.ok) {
        console.log('Experiment started successfully');
      } else {
        console.error('Failed to start experiment');
      }
    } catch (error) {
      console.error('Error starting experiment:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className='header'>Federated Learning Experiment Dashboard</h1>
      </header>
      
      <div class="divider"></div>

      <div className='Inputs'>
        <div class='strategy'>
          <h3 className='subheader'>Strategy</h3>
          <div class="select-wrapper">
            <select value={selectedStrategy} onChange={handleStrategyChange}>
              <option value="FedAvg">FedAvg</option>
              <option value="FedProx">FedProx</option>
              <option value="FedAdam">FedAdam</option>
              <option value="Krum">Krum</option>
              <option value="Bulyan">Bulyan</option>
            </select>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="none" d="M0 0h24v24H0z"/><path d="M7 10l5 5 5-5H7z"/></svg>
          </div>
        </div>


        <div className='dataset'>
          <h3 className='subheader'>Dataset</h3>
          <div class="select-wrapper">
            <select value={selectedDataset} onChange={handleDatasetChange}>
              <option value="CIFAR-10">CIFAR-10</option>
            </select>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="none" d="M0 0h24v24H0z"/><path d="M7 10l5 5 5-5H7z"/></svg>
          </div>
        </div>



        {/*TO DO: Align model preselect and file input*/}
        <div className='model'>
          <h3 className='subheader'>Model</h3>
          <div class="select-wrapper">
            <select value={selectedModel} onChange={handleModelChange}>
              <option value="CNN">CNN</option>
              <option value="ResNet">ResNet</option>
            </select>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="none" d="M0 0h24v24H0z"/><path d="M7 10l5 5 5-5H7z"/></svg>
          </div>
        </div>


        <div className='parameters'>
          <h3 className='subheader'>Hyperparameters</h3>
          <div className="parameter-inputs">
            <div className="parameter-input">
            <label htmlFor="learning-rate" style={{fontStyle: 'italic'}}>Learning Rate:</label>
              <input type="number" id="learning-rate" name="learning-rate" min="0" max="1" step="0.01" value={learningRate} onChange={handleLearningRateChange} />
            </div>
            <div className="parameter-input">
              <label htmlFor="momentum" style={{fontStyle: 'italic'}}>Momentum:</label>
              <input type="number" id="momentum" name="momentum" min="0" max="1" step="0.01" value={momentum} onChange={handleMomentumChange} />
            </div>
          </div>
        </div>
      </div>
      
      <div class="divider"></div>

      <div class="start">
        <div class="input-container">
          <h3 className='subheader' style={{marginRight: '10px'}}>Number of Rounds</h3>
          <input type="range" id="rounds" name="rounds" min="0" max="100" step="1" value={rounds} onChange={handleRoundsChange}/>
          <output id="rounds-value">{rounds}</output>
        </div>
        <div class="input-container">
          <h3 className='subheader' style={{marginRight: '10px'}}>Link Delay</h3>
          <input type="range" id="rounds" name="rounds" min="0" max="5000" step="1" value={linkDelay} onChange={handleLinkDelayChange}/>
          <output id="delay-value">{linkDelay}</output>
        </div>
        <div class="button-container">
          <button class="start-button" onClick={handleStartExperiment}>Start Experiment</button>
        </div>
      </div>

      <div class="divider"></div>

      <div className='output'>
        {/* Experiment log */}
        <div className="experiment-log">
          <h3 class='subheader'>Experiment Log</h3>
          {/* Display metrics data */}
          <div className='charts-container'>
            <div className='chart'>
              <Line options={accOptions} data={newAccuracyData} />
            </div>
            <div className='chart'>
              <Line options={lossOptions} data={newLossData} />
            </div>
          </div>
        </div>
      </div>

      <div class="divider"></div>

      <div class='table'>
        <table class='clean-table'>
          <thead>
            <tr>
              <th>Iteration</th>
              <th>Model</th>
              <th>Rounds</th>
              <th>Config</th>
              <th>Topology</th>
              <th>Logs</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>SimpleCNN</td>
              <td>10</td>
              <td>
                <button className='view-button'>Preview</button>
              </td>
              <td>
                <button className='view-button'>Open</button>
              </td>
              <td>
                <button className='view-button'>View</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="divider"></div>

      <h3 class='subheader'>Topology Visualization</h3>
      <div class="topology-visualization-div">
        <TopologyVisualization adjacencyList={adjacencyList} />
      </div>

      <div class="divider"></div>
    </div>
  );
}

export default App;
