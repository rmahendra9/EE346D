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
from utils.serializers import ndarrays_to_sparse_parameters
from utils.serializers import sparse_parameters_to_ndarrays
import datetime
from utils.scheduler import Optimal_Schedule
from utils.node_ip_mappings import generate_node_ip_mappings, get_node_info

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
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

def load_data(num_parts, is_iid, node_id):
    """Load partition CIFAR10 data."""
    if is_iid:
        part = IidPartitioner(num_parts)
    else:
        part = NumNodesGroupedNaturalIdPartitioner("label",num_groups=num_parts,num_nodes=num_parts)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": part})
    partition = fds.load_partition(node_id)
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

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--client-id",
    required=True,
    type=int,
    help="Partition of the dataset divided into iid partitions created artificially.",
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

client_id = args.client_id
num_clients = args.num_clients
model_type = args.model_type
is_iid=args.is_iid
node_id = client_id + 1
num_nodes = num_clients + 1

# Load model and data (simple CNN, CIFAR-10)
if model_type == 0:
    net = ResNet18().to(DEVICE)
else:
    net = SimpleCNN().to(DEVICE)

trainloader, testloader = load_data(num_parts=num_clients, is_iid=is_iid, node_id=client_id)


#Get schedule for node
num_chunks = 3
num_replicas = 1
num_segments = num_chunks*num_replicas
scheduler = Optimal_Schedule(num_nodes, num_segments, num_chunks, num_replicas)
schedule = scheduler.nodes_schedule[node_id]

#Get port to expose on this client
ip_mappings = generate_node_ip_mappings()
ip, port = get_node_info(node_id, ip_mappings)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, port):
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((socket.gethostname(), port))
        self.serversocket.listen(self.num_children)
        self.socket_open = False

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        #Train on params
        train(net, trainloader, epochs=1)
        send_to_server = False
        params_to_send_to_server = None 
        len_of_params = 0
        for communication in schedule:
            if communication['tx'] == 1:
                #Transmit chunk
                recv_node_id = communication['other_node']
                if recv_node_id != 0:
                    #Not communicating with server, so send params over tcp
                    recv_node_ip, recv_node_port = get_node_info(recv_node_id)
                    if self.socket_open:
                        self.socket.close()
                        self.socket_open = False
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket_open = True
                    sent = False
                    while not sent:
                        try:
                            #Attempt connection
                            print(f'Node {node_id} is attempting to connect to {recv_node_ip}:{recv_node_port}')
                            self.socket.connect((recv_node_ip, recv_node_port))
                            print(f'Node {node_id} successfully connected to {recv_node_ip}:{recv_node_port}')
                            #Send node_id
                            self.socket.send(str(node_id).encode())
                            ack = self.socket.recv(1024).decode()
                            print(f'Node {node_id} received ack from parent, will send data')

                            #TODO - send chunk of data
                            
                            #Send data
                            self.socket.sendall(data)
                            print(f"Node {node_id} sent data of size: {len(data)}")
                            sent = True
                        except Exception as e:
                            print(f'Unexpected {e=}, {type(e)=}')
                        finally:
                            sent = True
                    self.socket.close()
                    self.socket_open = False
                else:
                    #Make note to send chunk to server
                    send_to_server = True
                    params_to_send_to_server = self.get_parameters(config={})
                    len_of_params = len(self.get_parameters(config={}))
            else:
                #Receiving chunk
                #Accept connection
                (conn, addr) = self.serversocket.accept()
                #Get child node id
                child_node_id = conn.recv(1024).decode()
                print(f"Receiving data from node {child_node_id}")
                #Send acknowledgement
                conn.send('Ack'.encode())
                #Get current time to measure time delay of sending the data 
                start_time = datetime.datetime.now()
                #Receive the data
                data = []
                recv_params = []
                len_datasets = []
                while True:
                    packet = conn.recv(4096)
                    data.append(packet)
                    try:
                        pickle_data = b"".join(data)
                        data_arr = pickle.loads(pickle_data)
                        break
                    except pickle.UnpicklingError:
                        continue

                end_time = datetime.datetime.now()
                delay = end_time - start_time
                print(f'There is a delay of {delay.total_seconds()*1000} ms between this node and node {child_node_id}')

                #TODO - Post to API

                conn.shutdown(socket.SHUT_RDWR)

                data_arr = pickle.loads(pickle_data)
                recv_params.append(sparse_parameters_to_ndarrays(data_arr[0]))
                len_datasets.append(data_arr[1])

                print(f"Length of data received from node {child_node_id}: {len(pickle_data)}")
                conn.close()     
                #Aggregate parameters
                new_params = agg(recv_params, len_datasets)
                self.set_parameters(new_params)
                len_data = sum(len_datasets)
                #Serialize params
                ndarray_updated = self.get_parameters(config={})
                parameters_updated = ndarrays_to_sparse_parameters(ndarray_updated).tensors
        
        if send_to_server:
            return params_to_send_to_server, len_of_params, {}
        else:
            return self.get_parameters({}), 0, {}
            

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy, "loss": loss}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(port).to_client(),
)
