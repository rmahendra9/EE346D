import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,24, 3)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24,48, 3)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 24, 2)
        self.bn3 = nn.BatchNorm2d(24)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(17496,2000)
        self.fc2 = nn.Linear(2000,500)
        self.fc3 = nn.Linear(500,10)

    def forward(self, x):
        x = self.bn1(self.gelu(self.conv1(x)))
        x = self.bn2(self.gelu(self.conv2(x)))
        x = self.bn3(self.gelu(self.conv3(x)))

        x = self.flatten(x)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))

        return F.softmax(x)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
