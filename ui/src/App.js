import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className='App'>
      <div className='title'>
        <h1>Data Aggregation Networks for Federated Learning</h1>
      </div>
      <div className="dashboard-box">
        <h2>Model Outputs</h2>
        <div className="dashboard-item">
          <h4>Model Visualization</h4>
          {/* Add your model visualization component here */}
        </div>

        <div className="dashboard-item">
          <h4>Metrics</h4>
          {/* Add your log outputs component here */}
        </div>
        
        <div className="dashboard-item">
          <h4>Data Preview</h4>
          {/* Add your data preview component here */}
        </div>

        <div className="dashboard-item">
          <h4>Log Outputs</h4>
          {/* Add your log outputs component here */}
        </div>
      </div>
    </div>
  );
}

export default App;
