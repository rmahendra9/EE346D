import socket
import argparse
from scheduler import Optimal_Schedule
from logging import INFO 
from flwr.common.logger import log 

parser = argparse.ArgumentParser(description="Synchronizer")
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

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind((socket.gethostbyname(socket.gethostname()), 6000))
log(INFO, f'Synchronizer listening on port 6000')
serversocket.listen(num_nodes)


for i in range(num_rounds):
    scheduler = Optimal_Schedule(num_nodes, num_segments, num_chunks, num_replicas)
    total_slots = 0
    for i in range(len(scheduler.nodes_schedule)):
        for communication in scheduler.nodes_schedule[i]:
            total_slots = max(total_slots, communication['slot'])
    log(INFO, f'Found total number slots of {total_slots} for this round')
    for i in range(total_slots):
        log(INFO, f'Currently working on slot {i}')
        connections = []
        for j in range(num_nodes):
            (conn, addr) = serversocket.accept()
            connections.append(conn)
            node_id = conn.recv(1024).decode()
        for j in range(len(connections)):
            connections[j].send(str(i+1).encode())
            connections[j].close()
        log(INFO, f'Slot {i} is done')

serversocket.close()