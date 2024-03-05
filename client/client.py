import argparse
import warnings
from collections import OrderedDict
from utils.num_nodes_grouped_natural_id_partitioner import NumNodesGroupedNaturalIdPartitioner

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import socket
from models.ResNet import ResNet18
from models.simpleCNN import SimpleCNN

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


def load_data(node_id):
    """Load partition CIFAR10 data."""
    part = NumNodesGroupedNaturalIdPartitioner("label",num_groups=3,num_nodes=3)
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
    choices=[0, 1, 2, 3],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
node_id = parser.parse_args().node_id

parser.add_argument(
    "--parent_ip",
    required=False,
    type=str,
    help="IP address of this nodes parent"
)
parent_ip=parser.parse_args().parent_ip

parser.add_argument(
    "--parent_port",
    required=False,
    type=int,
    help="Port of this nodes parent"
)
parent_port=parser.parse_args().parent_port

parser.add_argument(
    "--has_parent",
    required=True,
    choices=[0,1],
    type=int,
    help="Denotes if node parent is a dual client"
)
has_parent=parser.parse_args().has_parent

# Load model and data (simple CNN, CIFAR-10)
#net = ResNet18().to(DEVICE)
net = SimpleCNN().to(DEVICE)

trainloader, testloader = load_data(node_id=node_id)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, parent_ip="0.0.0.0", parent_port=8080, has_parent=0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.has_parent = has_parent
        self.parent_ip = parent_ip
        self.parent_port = parent_port

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        #If node has parent as dual client, then it should send params to there
        if self.has_parent:
            self.socket.connect((self.parent_ip, self.parent_port))
            self.socket.send(self.get_parameters(config={}))
            #Return empty array to server, send params to parent
            return [], len(trainloader.dataset), {}
        #Otherwise, node is directly connected to main server, send to there
        else:
            return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(parent_ip, parent_port, has_parent).to_client(),
)
