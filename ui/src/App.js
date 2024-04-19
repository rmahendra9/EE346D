import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import "chart.js/auto";
import './App.css';
import TopologyVisualization from './TopologyVisualization';

function App() {
  const [selectedStrategy, setSelectedStrategy] = useState('FedAvg');
  const [selectedDataset, setSelectedDataset] = useState('CIFAR-10');
  const [selectedModel, setSelectedModel] = useState('CNN');
  const [learningRate, setLearningRate] = useState(0);
  const [momentum, setMomentum] = useState(0);
  const [newAccuracyData, setAccuracyData] = useState({
    labels: [],
    datasets: [{
      label: 'Model',
      data: [],
      color: 'blue',
      backgroundColor: 'blue'
    }]
  });
  const [newLossData, setLossData] = useState({
    labels: [],
    datasets: [{
      label: 'Model',
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

  const initialAdjacencyList = {
    A: [],
    B: [],
    C: []
  };

  const [adjacencyList, setAdjacencyList] = useState(initialAdjacencyList);
  const [newNodeName, setNewNodeName] = useState('');

  const handleAddNode = () => {
    if (newNodeName.trim() === '') {
      alert('Node name cannot be empty');
      return;
    }

    const newAdjacencyList = { ...adjacencyList, [newNodeName]: [] };
    setAdjacencyList(newAdjacencyList);
    setNewNodeName('');
  };


  const handleDeleteNode = (node) => {
    const newAdjacencyList = { ...adjacencyList };
    delete newAdjacencyList[node];
    for (let key in newAdjacencyList) {
      newAdjacencyList[key] = newAdjacencyList[key].filter((val) => val !== node);
    }
    setAdjacencyList(newAdjacencyList);
  };

  const handleCheckboxChange = (source, target, checked) => {
    const newAdjacencyList = { ...adjacencyList };
    if (checked) {
      if (!newAdjacencyList[source]) {
        newAdjacencyList[source] = [];
      }
      newAdjacencyList[source].push(target);
    } else {
      newAdjacencyList[source] = newAdjacencyList[source].filter((node) => node !== target);
    }
    setAdjacencyList(newAdjacencyList);
  };

  // Function to fetch metrics and update state
  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:80/metrics');
      const data = await response.json();
      if (data) {
        let labelModel = 'CNN';
        if(data.length > 0 && data[0].model == 'ResNet') {
          console.log('Reached');
          labelModel = 'ResNet';
        }
        const newAccuracyData = {
          labels: [],
          datasets: [{
            label: labelModel,
            data: [],
            color: 'blue',
            backgroundColor: 'blue'
          }]
        };
        const newLossData = {
          labels: [],
          datasets: [{
            label: labelModel,
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
    }, 1000); // Fetch every 0.8 seconds (adjust as needed)

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
          num: 0
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
        <div class="button-container">
          <button class="start-button" onClick={handleStartExperiment}>Start Experiment</button>
          {/* TO DO: Loading button functionality to indicate duration of experiment */}
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

      {/* TO DO: Add log file server output here --> different branch */}

      {/*<div class='table'>
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

      <div class="divider"></div>*/}

      <h3 class='subheader'>Topology Visualization</h3>
      <div class="topology-visualization-div">
        <TopologyVisualization adjacencyList={adjacencyList} />
      </div>
      <div class='add-node'>
        <h4>Add Node</h4>
        <input
          type="text"
          placeholder="Node Name"
          value={newNodeName}
          onChange={(e) => setNewNodeName(e.target.value)}
        />
        <button onClick={handleAddNode}>Add Node</button>
      </div>
      <div class='adjacency-matrix'>
        <h4>Adjacency Matrix</h4>
        <table>
          <thead>
            <tr>
              <th></th>
              {Object.keys(adjacencyList).map(node => (
                <th key={node}>{node}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.keys(adjacencyList).map(source => (
              <tr key={source}>
                <td>{source}</td>
                {Object.keys(adjacencyList).map(target => (
                  <td key={`${source}-${target}`}>
                    <input
                      type="checkbox"
                      checked={adjacencyList[source].includes(target)}
                      onChange={(e) => handleCheckboxChange(source, target, e.target.checked)}
                    />
                  </td>
                ))}
                <td><button onClick={() => handleDeleteNode(source)}>Delete Node</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div class="divider"></div>
    </div>
  );
}

export default App;
