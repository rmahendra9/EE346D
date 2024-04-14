import argparse
import warnings
from collections import OrderedDict
from utils.num_nodes_grouped_natural_id_partitioner import NumNodesGroupedNaturalIdPartitioner

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
from models.ResNet import ResNet18
from models.simpleCNN import SimpleCNN
from utils.serializers import ndarrays_to_sparse_parameters
import pickle
import datetime

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
    "--node-id",
    required=True,
    type=int,
    help="Partition of the dataset divided into iid partitions created artificially.",
)


parser.add_argument(
    "--parent_ip",
    required=False,
    type=str,
    help="IP address of the parent"
)


parser.add_argument(
    "--parent_port",
    required=False,
    type=int,
    help="Port exposed on the parent"
)


parser.add_argument(
    "--is_parent_dual",
    required=True,
    choices=[0,1],
    type=int,
    help="Denotes if node parent is a dual client"
)

parser.add_argument(
    "--num_nodes",
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

is_parent_dual= args.is_parent_dual
parent_port=args.parent_port
parent_ip=args.parent_ip
node_id = args.node_id
num_nodes = args.num_nodes
model_type = args.model_type
is_iid=args.is_iid

# Load model and data (simple CNN, CIFAR-10)
if model_type == 0:
    net = ResNet18().to(DEVICE)
else:
    net = SimpleCNN().to(DEVICE)

trainloader, testloader = load_data(num_parts=num_nodes, is_iid=is_iid, node_id=node_id)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, parent_ip, parent_port, is_parent_dual):
        
        self.is_parent_dual = is_parent_dual
        self.parent_ip = parent_ip
        self.parent_port = parent_port
        self.socket_open = False


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        #Train locally
        train(net, trainloader, epochs=1)
        #Serialize parameters
        ndarray_updated = self.get_parameters(config={})
        parameters_updated = ndarrays_to_sparse_parameters(ndarray_updated).tensors
        #Parent is dual client, so need to send params there
        if self.is_parent_dual:
            if self.socket_open:
                self.socket.close()
                self.socket_open = False
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_open = True
            sent = False
            while not sent:
                try:
                    #Attempt connection
                    print(f'Node {node_id} is attempting to connect to {self.parent_ip}:{self.parent_port}')
                    self.socket.connect((self.parent_ip, self.parent_port))
                    print(f'Node {node_id} successfully connected to {self.parent_ip}:{self.parent_port}')
                    #Send node id
                    self.socket.sendall(str(node_id).encode())
                    #Prepare data to send
                    data = pickle.dumps([parameters_updated,len(trainloader.dataset)])
                    #Send timestamp before sending data
                    self.socket.sendall(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").encode())
                    #Send data
                    self.socket.sendall(data)
                    print(f"Node {node_id} sent data of size: {len(data)}")
                    sent = True
                except Exception as e:
                    print(f'Unexpected {e=}, {type(e)=}')
                finally:
                    sent = True
            return self.get_parameters({}), 0, {}
            #Return empty array to server, send params to parent
            
        #Otherwise, node is directly connected to main server, send to there
        else:
            return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy, "loss": loss}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(parent_ip, parent_port, is_parent_dual).to_client(),
)