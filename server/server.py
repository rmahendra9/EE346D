import flwr as fl
import wandb
import argparse
import datetime
from typing import List, Tuple, Optional, Callable, Dict
from flwr.common import Metrics
from logging import INFO 
from flwr.common.logger import log
from pathlib import Path
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from strategy import CustomFed
from scheduler import Optimal_Schedule
import pickle

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
parser.add_argument(
    "--num_chunks",
    required=True,
    type=int,
    help="Number of chunks to split files into"
)
parser.add_argument(
    "--num_replicas",
    required=True,
    type=int,
    help="Number of replicas of each chunk"
)
args = parser.parse_args()
num_rounds = args.num_rounds
num_nodes = args.num_nodes
num_chunks = args.num_chunks
num_replicas = args.num_replicas
num_segments = num_chunks*num_replicas

wandb.init(
    project = 'FLNET',
    name  = 'FLNET'
)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    metric = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    log(INFO, f'{metric}')
    return metric


#Config for fit function
def get_on_fit_config_fn() -> Callable[[int], Dict[str, bytes]]:
    """Return a function which returns training configurations."""
    def fit_config(server_round: int) -> Dict[str, bytes]:
        """Return training configuration dict for each round."""
        config = {}
        #Create scheduler instance
        scheduler = Optimal_Schedule(num_nodes, num_segments, num_chunks, num_replicas)
        schedule = {}
        
        #clear schedule file
        pickle.dump(scheduler.nodes_schedule, open('schedule.pkl','wb'))
        
        #Set schedule for each node
        for i in range(len(scheduler.nodes_schedule)):
            config[str(i)] = pickle.dumps(scheduler.nodes_schedule[i])

        #Send other information
        config['server_round'] = pickle.dumps(server_round)
        config['num_chunks'] = pickle.dumps(num_chunks)
        config['num_replicas'] = pickle.dumps(num_replicas)
        #Send total number of communication slots
        total_slots = 0
        for i in range(len(scheduler.nodes_schedule)):
            for communication in scheduler.nodes_schedule[i]:
                total_slots = max(total_slots, communication['slot'])
        config['total_slots'] = pickle.dumps(total_slots)
        return config
    return fit_config

# Define strategy - TODO (We need to be able to support more strategies)
strategy = CustomFed(evaluate_metrics_aggregation_fn=weighted_average,
                     on_fit_config_fn=get_on_fit_config_fn(),
                    )

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

#Clean out log file
open('log.txt','w').close()

open('schedule.txt','w').close()





# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=ClientManager(num_nodes),
)