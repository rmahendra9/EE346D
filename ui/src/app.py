from flask import Flask, render_template, jsonify
from typing import List, Tuple
import json
from flwr.common import Metrics
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

METRICS_FILE = "metrics.json"

# Load initial metrics from the file
try:
    with open(METRICS_FILE, "r") as file:
        metricList = json.load(file)
except FileNotFoundError:
    metricList = []

@app.route("/")
def index():
    return render_template("index.html")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by the number of examples used
    print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    print('GETTING METRICS')
    
    # Aggregate and return custom metric (weighted average)
    metric = {"accuracy": sum(accuracies) / sum(examples)}
    metricList.append(metric)
    
    # Save the updated metrics to the file
    with open(METRICS_FILE, "w") as file:
        json.dump(metricList, file)
    
    return metricList[-1]  # Return the latest metric as a dictionary

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        with open(METRICS_FILE, "r") as file:
            metricList = json.load(file)
    except FileNotFoundError:
        metricList = []
    return metricList


if __name__ == "__main__":
    app.run(debug=True, port=5000)
