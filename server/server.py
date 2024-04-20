import flwr as fl
import wandb
import argparse
import datetime
from typing import List, Tuple, Optional
from flwr.common import Metrics
from logging import INFO 
from flwr.common.logger import log
from pathlib import Path
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from strategy import CustomFed

#TODO - strategy stuff - diff strats, etc

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--num_rounds",
    required=True,
    type=int,
    help="Number of rounds for FL experiment",
)
parser.add_argument(
    "--num_nodes",
    required=True,
    type=int,
    help="Number of nodes in the FL experiment",
)
args = parser.parse_args()
num_rounds = args.num_rounds
num_nodes = args.num_nodes

wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)


#TODO - Do we need this for fit also? Also do we only want weighted average ?
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    metric = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    log(INFO, f'{metric}')
    return metric


#Config for fit function
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round
    }
    return config

# Define strategy - TODO (We need to be able to support more strategies)
strategy = CustomFed(evaluate_metrics_aggregation_fn=weighted_average,on_fit_config_fn=fit_config)

#Configure logging
fl.common.logger.configure(identifier="Federated_Learning", filename="log.txt")

#Set up client manager to 
class ClientManager(SimpleClientManager):
    def __init__(self, min_clients) -> None:
        super().__init__()
        self.min_clients = min_clients
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = 1,
        criterion = None,
    ) -> List[ClientProxy]:
        return super().sample(num_clients,self.min_clients,criterion)





# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=ClientManager(num_nodes),
)