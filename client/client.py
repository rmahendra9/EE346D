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
from models.ResNet import ResNet18, ResNet50, ResNet34, ResNet101
from models.simpleCNN import SimpleCNN
import datetime
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

# LOAD THE DATA FROM THE SERVER SIDE (SCHEDULER)
# WE ARE GOING TO RECEIVE
# OUR ARGS
# NODE_ID, NUM_CLIENTS, MODEL_TYPE, IS_IID

# AND THE NODE ID MAPPINGS
# SYNCHRONIZER IP, SERVER IP, (NODE_ID, NODE_IP)

PORT = 3002
SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOCKET.bind((socket.gethostbyname(socket.gethostname()), PORT))
SOCKET.listen(1)

print(f'{socket.gethostbyname(socket.gethostname())}')
(conn, _) = SOCKET.accept()


data = []
#Get current time to measure time delay of sending the data 
start_time = datetime.datetime.now()
while True:
    packet = conn.recv(16384)
    data.append(packet)
    try:
        pickle_data = b"".join(data)
        config = pickle.loads(pickle_data)
        break
    except pickle.UnpicklingError:
        continue

#SOCKET.close()

node_id = config['node_id']
num_nodes = config['num_nodes'] 
model_type = config['model_type']
is_iid = config['is_iid']
num_clients = num_nodes - 1
client_id = node_id - 1
ip_mappings = config['mappings']
server_ip = config['server_ip']
server_port = config['server_port']
synchronizer_node_ip = config['synchronizer_ip']
synchronizer_node_port = config['synchronizer_port']

# ip_mappings -> list where ip of node i is in position i
# [(0.0.0.0, 80), (1.1.1.1,90), (2.2.2.2,30)]

# Load model and data (simple CNN, CIFAR-10)
if model_type == 0:
    net = SimpleCNN().to(DEVICE)
elif model_type ==1:
    net = ResNet18().to(DEVICE)
elif model_type ==2:
    net = ResNet34().to(DEVICE)
elif model_type ==3:
    net = ResNet50().to(DEVICE)
elif model_type == 4:
    net = ResNet101().to(DEVICE)

if node_id != 0:
    trainloader, testloader = load_data(num_parts=num_clients, is_iid=is_iid, client_id=client_id)

#ip_mappings = generate_node_ip_mappings(num_nodes)
#ip, port = get_node_info(node_id, ip_mappings)
ip, port = ip_mappings[node_id]
#synchronizer_node_ip = '127.0.1.1'
#synchronizer_node_port = 6000

fl.common.logger.configure(identifier="Federated_Learning", filename="log.txt")
open('log.txt','w').close()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, port):
        self.serversocket = SOCKET
        #self.serversocket.bind((socket.gethostbyname(socket.gethostname()), port))
        self.serversocket.listen(num_nodes)
        self.socket_open = False
        self.net = net
    
    def __del__(self):
        self.serversocket.close()

    def get_parameters(self, config=None):
        return self.net.get_parameters()

    def set_parameters(self, parameters):
        self.net.set_parameters(parameters)
        self.net = self.net.float().to(DEVICE)

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
        #Set parameters
        self.set_parameters(parameters)
        #Train on params if not server
        if node_id != 0:
            train(net, trainloader, epochs=1)
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
                try:
                    #Close socket
                    if self.socket_open:
                        self.socket.close()
                        self.socket_open = False
                    #Create socket
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.socket_open = True
                    self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                    #Send node id
                    self.socket.send(str(node_id).encode())
                    slot_id = int(self.socket.recv(1024).decode())
                    self.socket.close()
                    self.socket_open = False
                except Exception as e:
                    log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
                    time.sleep(1)
            else:
                #Talk in current slot id
                if schedule[communication_idx]['tx'] == 1:
                    #Transmit chunk
                    chunk_id = schedule[communication_idx]['segment']
                    recv_node_id = schedule[communication_idx]['other_node']
                    recv_node_ip, recv_node_port = ip_mappings[recv_node_id]
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
                            self.socket.connect((recv_node_ip, recv_node_port))
                            #Send node_id and chunk_id
                            self.socket.send(f'{node_id}:{chunk_id}'.encode())
                            ack = self.socket.recv(1024).decode()

                            #Create chunk of data
                            chunk = chunks[chunk_id]

                            #Serialize the chunk
                            data = pickle.dumps([chunk,len_datasets[chunk_id]])
                            
                            #Send data
                            self.socket.sendall(data)
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
                        self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                        #Send node id
                        self.socket.send(str(node_id).encode())
                        slot_id = int(self.socket.recv(1024).decode())
                        self.socket.close()
                        self.socket_open = False
                    except Exception as e:
                        log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
                else:
                    #Receiving chunk
                    #Accept connection
                    (conn, addr) = self.serversocket.accept()
                    #Get child node id
                    msg = conn.recv(1024).decode()
                    info = msg.split(":")
                    chunk_id = int(info[1])
                    child_node_id = int(info[0])
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
                    log(INFO, f'There is a delay of {delay.total_seconds()*1000:.3f} ms between this node and node {child_node_id} in communicating chunk {chunk_id}')
                    #Load data
                    data_arr = pickle.loads(pickle_data)
                    #Aggregate parameters using weighted average
                    new_params = data_arr[0]
                    len_datasets_chunk = data_arr[1]
                    chunks[chunk_id] = ((np.array(new_params)* len_datasets_chunk + np.array(chunks[chunk_id])* len_datasets[chunk_id])/(len_datasets_chunk + len_datasets[chunk_id])).tolist()

                    len_datasets[chunk_id] += len_datasets_chunk

                    conn.close()     

                    len_data = max(len_datasets)
                    try:
                        #Now send ack to synchronizer and recv slot_id
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.socket_open = True
                        self.socket.connect((synchronizer_node_ip, synchronizer_node_port))
                        #Send node id
                        self.socket.send(str(node_id).encode())
                        slot_id = int(self.socket.recv(1024).decode())
                        self.socket.close()
                        self.socket_open = False
                    except Exception as e:
                        log(INFO, f'Unexpected {e=}, {type(e)=}, with message {repr(e)} from node {node_id}')
                #Increment communication id
                communication_idx += 1
        #Set model parameters
        model = restore_weights_from_flat(net, chunks)
        if node_id == 0:
            #Send to server
            return model.get_parameters(), len_data, {}
        else:
            #Send null to server
            return [], 0, {}

    def evaluate(self, parameters, config):
        #Client nodes send loss and accuracy to server
        if node_id != 0:
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return loss, len(testloader.dataset), {"accuracy": accuracy, "loss": loss}
        #"Server node" should not send anything, be ignored
        else:
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}




# LOAD DATA


# Start Flower client
fl.client.start_client(
    server_address=f"{server_ip}:{server_port}",
    client=FlowerClient(port).to_client(),
)
