import flwr as fl
from app import weighted_average
import wandb
# Define strategy
import argparse

"""parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--strategy",
    choices=[0, 1, 2, 3],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)"""

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)