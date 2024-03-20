import flwr as fl
from app import weighted_average
import wandb
# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:443",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)