import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_parameters(self):
         return [param.data.cpu().numpy() for param in self.parameters()]
    
    def set_parameters(self, parameters):
        i=0
        for param in self.parameters():
            param.data = torch.tensor(parameters[i])
            i += 1