import argparse
import warnings
from collections import OrderedDict
from utils.num_nodes_grouped_natural_id_partitioner import NumNodesGroupedNaturalIdPartitioner
import pickle

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner.iid_partitioner import IidPartitioner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import socket
import numpy as np
from models.ResNet import ResNet18
from models.simpleCNN import SimpleCNN
from models.CustomCNN import CustomCNN
import datetime
from utils.node_ip_mappings import generate_node_ip_mappings, get_node_info
from logging import INFO 
from flwr.common.logger import log 
from pathlib import Path
from utils.chunker import get_flattened_weights, split_list, restore_weights_from_flat
import time

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net = net.float()
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
    return net

def test(net, testloader):
    """Validate the model on the test set."""
    net = net.float()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

#TODO - is this the only agg function we want?
def agg(param_list, len_datasets):
    final_params = []
    for i in range(len(param_list[0])):
        final_params.append(np.mean(np.array([param_list[j][i] for j in range(len(param_list))]), axis=0))

    return final_params

def load_data(num_parts, is_iid, client_id):
    """Load partition CIFAR10 data."""
    if is_iid:
        part = IidPartitioner(num_parts)
    else:
        part = NumNodesGroupedNaturalIdPartitioner("label",num_groups=num_parts,num_nodes=num_parts)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": part})
    partition = fds.load_partition(client_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node_id",
    required=True,
    type=int,
    help="ID of the current node",
)

parser.add_argument(
    "--num_clients",
    required=True,
    type=int,
    help="Number of nodes in the federated learning"
)

parser.add_argument(
    "--model_type",
    required=True,
    type=int,
    choices=[0,1],
    help="Select model type - 0 for ResNet18, 1 for SimpleCNN"
)

parser.add_argument(
    "--is_iid",
    required=True,
    type=int,
    choices=[0,1],
    help="Denotes if data is iid - 0 for no, 1 for yes"
)

args = parser.parse_args()

node_id = args.node_id
num_clients = args.num_clients
model_type = args.model_type
is_iid=args.is_iid
num_nodes = num_clients + 1
client_id = node_id - 1

# Load model and data (simple CNN, CIFAR-10)
if model_type == 0:
    net = ResNet18().to(DEVICE)
    #net = CustomCNN().to(DEVICE)
else:
    net = SimpleCNN().to(DEVICE)

if node_id != 0:
    trainloader, testloader = load_data(num_parts=num_clients, is_iid=is_iid, client_id=client_id)

ip_mappings = generate_node_ip_mappings(num_nodes)
ip, port = get_node_info(node_id, ip_mappings)
synchronizer_node_ip = '10.52.3.223'
synchronizer_node_port = 6000

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, port):
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((socket.gethostname(), port))
        log(INFO, f'Node {node_id} listening on port {port}')
        self.serversocket.listen(num_nodes)
        self.socket_open = False
        self.net = net

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).float() for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        #Get server round
        current_round = pickle.loads(config['server_round'])
        #Get schedule
        schedule = pickle.loads(config[str(node_id)])
        #Get num_chunks and num_replicas
        num_chunks = pickle.loads(config['num_chunks'])
        num_replicas = pickle.loads(config['num_replicas'])
        total_slots = pickle.loads(config['total_slots'])
        #Sort communications by slot id
        schedule = sorted(schedule, key=lambda communication: communication['slot'])
        #Print schedule for the round
        log(INFO, f'Node {node_id} schedule for round {current_round}: {schedule}')
        #Set parameters
        self.set_parameters(parameters)
        #Train on params if not server
        if node_id != 0:
            self.net = train(self.net, trainloader, epochs=1)
        #Generate chunks from parameters
        chunks = split_list(get_flattened_weights(self.get_parameters(config={})), num_chunks)
        #Length of the dataset for each chunk
        if node_id != 0:
            len_datasets = [len(trainloader.dataset)]*num_chunks
        else:
            len_datasets = [0]*num_chunks
        #Maximum size dataset for a chunk
        len_data = 0
        #Set slot id
        slot_id = 0
        #Set index for communication
        communication_idx = 0
        #Find max slot id in schedule
        max_slot = 0
        for i in range(len(schedule)):
            max_slot = max(max_slot, schedule[i]['slot'])
        while slot_id < total_slots:
            if slot_id > max_slot or slot_id != schedule[communication_idx]['slot']:
                #Not current slot id, send ack to synchronizer and wait for next slot id
                log(INFO, f'Node {node_id} is not communicating in slot {slot_id}')
                try:
                    #Close socket
                    if self.socket_open:
                        self.socket.close()
                        self.socket_open = False
                    #Create socket
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.socket_open = True
                    log(INFO, f'Node {node_id} is attempting to connect to {synchronizer_node_ip}:{synchronizer_node_port}')
                    self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                    log(INFO, f'Node {node_id} successfully connected to {synchronizer_node_ip}:{synchronizer_node_port}')
                    #Send node id
                    self.socket.send(str(node_id).encode())
                    slot_id = int(self.socket.recv(1024).decode())
                    log(INFO, f'Received slot {slot_id} from synchronizer node')
                    self.socket.close()
                    self.socket_open = False
                except Exception as e:
                    log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
            else:
                #Talk in current slot id
                if schedule[communication_idx]['tx'] == 1:
                    log(INFO, f'Node {node_id} is transmitter for slot {slot_id}')
                    #Transmit chunk
                    chunk_id = schedule[communication_idx]['segment']
                    recv_node_id = schedule[communication_idx]['other_node']
                    recv_node_ip, recv_node_port = get_node_info(recv_node_id, ip_mappings)
                    log(INFO, f'Node {node_id} will send chunk {chunk_id} to node {recv_node_id}')

                    #Just to be safe, close the socket
                    if self.socket_open:
                        self.socket.close()
                        self.socket_open = False
                    #Create socket
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.socket_open = True
                    sent = False
                    while not sent:
                        try:
                            #Attempt connection
                            log(INFO, f'Node {node_id} is attempting to connect to {recv_node_ip}:{recv_node_port}')
                            self.socket.connect((recv_node_ip, recv_node_port))
                            log(INFO, f'Node {node_id} successfully connected to {recv_node_ip}:{recv_node_port}')

                            #Send node_id and chunk_id
                            self.socket.send(f'{node_id}:{chunk_id}'.encode())
                            ack = self.socket.recv(1024).decode()
                            log(INFO, f'Node {node_id} received ack from parent, will send chunk {chunk_id}')

                            #Create chunk of data
                            chunk = chunks[chunk_id]

                            #Serialize the chunk
                            data = pickle.dumps([chunk,len_datasets[chunk_id]])
                            
                            #Send data
                            self.socket.sendall(data)
                            log(INFO, f"Node {node_id} sent data of size: {len(data)}")
                            sent = True

                        except Exception as e:
                            log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')

                    self.socket.close()
                    self.socket_open = False
                    try:
                        #Now send ack to synchronizer and recv slot_id
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.socket_open = True
                        log(INFO, f'Node {node_id} is attempting to connect to {synchronizer_node_ip}:{synchronizer_node_port}')
                        self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                        log(INFO, f'Node {node_id} successfully connected to {synchronizer_node_ip}:{synchronizer_node_port}')
                        #Send node id
                        self.socket.send(str(node_id).encode())
                        slot_id = int(self.socket.recv(1024).decode())
                        log(INFO, f'Received slot {slot_id} from synchronizer node')
                        self.socket.close()
                        self.socket_open = False
                    except Exception as e:
                        log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
                else:
                    #Receiving chunk
                    log(INFO, f'Node {node_id} is receiver for slot {slot_id}')
                    #Accept connection
                    (conn, addr) = self.serversocket.accept()
                    #Get child node id
                    msg = conn.recv(1024).decode()
                    info = msg.split(":")
                    chunk_id = int(info[1])
                    child_node_id = int(info[0])
                    log(INFO, f"Node is receiving chunk {chunk_id} from node {child_node_id}")
                    #Send acknowledgement
                    conn.send('Ack'.encode())
                    #Receive the data
                    data = []
                    #Get current time to measure time delay of sending the data 
                    start_time = datetime.datetime.now()
                    while True:
                        packet = conn.recv(16384)
                        data.append(packet)
                        try:
                            pickle_data = b"".join(data)
                            data_arr = pickle.loads(pickle_data)
                            break
                        except pickle.UnpicklingError:
                            continue
                    #Get current time to measure time delay
                    end_time = datetime.datetime.now()
                    delay = end_time - start_time
                    log(INFO, f'There is a delay of {delay.total_seconds()*1000} ms between this node and node {child_node_id}')
                    #Load data
                    data_arr = pickle.loads(pickle_data)
                    #Aggregate parameters using weighted average
                    new_params = data_arr[0]
                    len_datasets_chunk = data_arr[1]
                    chunks[chunk_id] = ((np.array(new_params)* len_datasets_chunk + np.array(chunks[chunk_id])* len_datasets[chunk_id])/(len_datasets_chunk + len_datasets[chunk_id])).tolist()

                    len_datasets[chunk_id] += len_datasets_chunk

                    log(INFO, f"Length of data received from node {child_node_id}: {len(pickle_data)}")
                    conn.close()     

                    len_data = max(len_datasets)
                    try:
                        #Now send ack to synchronizer and recv slot_id
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.socket_open = True
                        log(INFO, f'Node {node_id} is attempting to connect to {synchronizer_node_ip}:{synchronizer_node_port}')
                        self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                        log(INFO, f'Node {node_id} successfully connected to {synchronizer_node_ip}:{synchronizer_node_port}')
                        #Send node id
                        self.socket.send(str(node_id).encode())
                        slot_id = int(self.socket.recv(1024).decode())
                        log(INFO, f'Received slot {slot_id} from synchronizer node')
                        self.socket.close()
                        self.socket_open = False
                    except Exception as e:
                        log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
                #Increment communication id
                communication_idx += 1
        #Set model parameters
        self.net = restore_weights_from_flat(self.net, chunks)
        if node_id == 0:
            #Send to server
            return self.get_parameters(), len_data, {}
        else:
            #Send null to server
            return [], 0, {}

    def evaluate(self, parameters, config):
        #Client nodes send loss and accuracy to server
        if node_id != 0:
            self.set_parameters(parameters)
            loss, accuracy = test(self.net, testloader)
            return loss, len(testloader.dataset), {"accuracy": accuracy, "loss": loss}
        #"Server node" should not send anything, be ignored
        else:
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}


# Start Flower client
fl.client.start_client(
    server_address="10.52.3.223:8080",
    client=FlowerClient(port).to_client(),
)
