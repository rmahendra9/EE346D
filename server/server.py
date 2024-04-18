import flwr as fl
import wandb
import argparse
import datetime
from typing import List, Tuple
from flwr.common import Metrics

#TODO - strategy stuff - diff strats, etc

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--num_rounds",
    required=True,
    type=int,
    help="Number of rounds for FL experiment",
)
parser.add_argument(
    "--num_clients",
    required=True,
    type=int,
    help="Number of clients in FL experiment"
)
args = parser.parse_args()
num_rounds = args.num_rounds
num_clients = args.num_clients

wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)

prev_time = datetime.datetime.now()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    #Calculate round delay
    global prev_time
    curr_time = datetime.datetime.now()
    round_delay = curr_time - prev_time
    print(f'There was a round delay of {round_delay.total_seconds()} seconds for this round')
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    metric = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    print(metric)
    prev_time = datetime.datetime.now()
    return metric

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

#Configure logging
fl.common.logger.configure(identifier="Federated_Learning", filename="log.txt")

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)