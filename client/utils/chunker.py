import numpy as np
import torch
"""
Contains the function to generate the chunks and restore the model
"""
def get_flattened_weights(parameters) -> np.array:
    return np.concatenate([x_.flatten() for x_ in parameters])

def get_shapes(parameters):
    return [len(x_.flatten()) for x_ in parameters]

def get_cum_sum(parameters):
    return np.array([0] + list(np.cumsum(get_shapes(parameters))))

def restore_weights_from_flat(model, flattened_weights):
    lens = get_cum_sum(model.get_parameters())
    splitted = [flattened_weights[lens[i]:lens[i+1]] for i in range(len(lens)-1)]
    for i,param in enumerate(model.parameters()):
        param.data = torch.from_numpy(splitted[i].reshape(param.data.shape))



def split_list(lst, n):
    avg = len(lst) / float(n)
    parts = []
    last = 0.0

    while last < len(lst):
        parts.append(lst[int(last):int(last + avg)])
        last += avg

    return parts

def get_chunk(params, num_chunks, chunk_id):
    weights = get_flattened_weights(params)
    print(len(weights))
    return split_list(weights, num_chunks)[chunk_id]