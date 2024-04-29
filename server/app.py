from flask import Flask, render_template, jsonify, request
from typing import List, Tuple
import json
from flwr.common import Metrics
from flask_cors import CORS
import wandb
import numpy as np
import subprocess
import os
import pickle

app = Flask(__name__)
CORS(app)

METRICS_FILE = "metrics.json"
LOGS_FILE = "log.txt"
SCHEDULE_FILE = "schedule.pkl"

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
    #print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    models = [num_examples * m["model"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    model = ""
    if "CNN" in models[0]:
        model = "CNN"
    else:
        model = "ResNet"
    
    # Aggregate and return custom metric (weighted average)
    metric = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "model": model}
    print(metric)
    #wandb.log(metric)
    metricList.append(metric)
    
    # Save the updated metrics to the file
    with open(METRICS_FILE, "w") as file:
        json.dump(metricList, file)
    
    return metricList[-1]  # Return the latest metric as a dictionary

@app.route('/start-experiment', methods=['POST'])
def start_experiment():
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)

    if os.path.exists(LOGS_FILE):
        os.remove(LOGS_FILE)

    # File is accessible from request.files
    file = request.files.get('file')
    if file:
        # Save or process the file
        file.save((os.path.join('../client/scheduler', file.filename)))

    # Access other form data from request.form (not request.json)
    strModel = request.form.get('model')
    model = '0'
    if strModel == "CNN":
        model = '1'

    is_iid = request.form.get('iid')

    ipList = request.form.get('ipList')
    ipList = ipList.split(',')
    numNodes = len(ipList) + 1
    clients = numNodes - 1

    chunks = int(request.form.get('chunks'))
    numRounds = 10

    subprocess.Popen(['python3', 'server.py', '--num_rounds', str(numRounds), '--num_nodes', str(numNodes), '--num_chunks', str(chunks), '--num_replicas', '1'])
    subprocess.Popen(['python3', 'synchronizer.py', '--num_rounds', str(numRounds), '--num_nodes', str(numNodes), '--num_chunks', str(chunks), '--num_replicas', '1'])

    startId = 0
    while numNodes > 0:
        subprocess.Popen(['python3', '../client/client.py', '--node_id', str(startId), '--num_clients', str(clients), '--model_type', str(model), '--is_iid', str(is_iid)])
        numNodes -= 1
        startId += 1

    return jsonify({'message': 'Experiment started successfully'})

@app.route('/upload', methods=['POST'])
def file_upload():
    # Check if a file is sent to the route
    if 'file' in request.files:
        file = request.files['file']
        # You can now save the file, process it, etc.
        file.save('/path/to/save/' + file.filename)  # Save the file

        return jsonify({'message': 'File uploaded successfully!'}), 200
    else:
        return jsonify({'error': 'No file provided'}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        with open(METRICS_FILE, "r") as file:
            metricList = json.load(file)
    except FileNotFoundError:
        metricList = []
    return metricList

@app.route('/logs', methods=['GET'])
def get_logs():
    log_list = []
    try:
        with open("log.txt", "r") as file:
            for line in file:
                log_list.append(line)
    except FileNotFoundError:
        log_list = []
    return jsonify(log_list)

@app.route('/schedule', methods=['GET'])
def get_schedule():
    schedule_list = []
    try:
        schedule_list = pickle.load(open(SCHEDULE_FILE,'rb'))
        return schedule_list
    except FileNotFoundError:
        schedule_list = {}
    return schedule_list


if __name__ == "__main__":
    app.run(debug=True, port=80)