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
    flattened_weights = np.array(flattened_weights).flatten().tolist()
    lens = get_cum_sum(model.get_parameters())
    splitted = [flattened_weights[lens[i]:lens[i+1]] for i in range(len(lens)-1)]
    i = 0
    for param in model.parameters():
        param.data = torch.from_numpy(np.array(splitted[i]).reshape(param.data.shape))
        i += 1
        
    return model



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
    return split_list(weights, num_chunks)[chunk_id]