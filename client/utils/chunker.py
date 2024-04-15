import numpy as np
import torch
"""
Contains the function to generate the chunks and restore the model
"""
def get_flattened_weights(model) -> np.array:
    return np.concatenate([x_.data.flatten().numpy() for x_ in model.parameters()])

def get_shapes(model):
    return [len(x_.data.flatten().numpy()) for x_ in model.parameters()]

def get_cum_sum(model):
    return np.array([0] + list(np.cumsum(get_shapes(model))))

def restore_weights_from_flat(model, flattened_weights):
    lens = get_cum_sum(model)
    splitted = [flattened_weights[lens[i]:lens[i+1]] for i in range(len(lens)-1)]
    i = 0
    for param in model.parameters():
        param.data = torch.from_numpy(splitted[i].reshape(param.data.shape))
        i += 1