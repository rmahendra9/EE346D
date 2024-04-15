import flwr as fl
from app import weighted_average
import wandb
import argparse

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

#TODO - strategy stuff - diff strats, etc

#TODO - assign IPs to node ids 

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--num_rounds",
    required=True,
    type=int,
    help="Number of rounds for FL experiment",
)
args = parser.parse_args()
num_rounds = args.num_rounds

wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)