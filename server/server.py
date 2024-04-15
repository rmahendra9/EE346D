import flwr as fl
from app import weighted_average
import wandb
import argparse
from flwr.server.client_manager import SimpleClientManager
import random
import threading
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
class ClientManager(SimpleClientManager):
    def __init__(self, min_clients) -> None:
        super().__init__()
        self.min_clients = min_clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = 2,
        criterion = None,
    ) -> List[ClientProxy]:
        return super().sample(num_clients,self.min_clients,criterion)

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

parser.add_argument(
    "--min_clients",
    default=3,
    required=False,
    type=int,
    help="The minimum number of training clients"
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
    client_manager= ClientManager(3)
)


