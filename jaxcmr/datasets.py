__all__ = ['load_parameters', 'load_embeddings', 'load_data', 'generate_trial_mask']

import numpy as np
import hdf5storage
import json


def load_parameters(parameter_path: str):
    with open(parameter_path, 'r') as f:
        parameters = json.load(f)
    return parameters


def load_embeddings(embedding_paths):
    """Load embeddings from npy files."""
    embeddings = []
    for i in range(len(embedding_paths)):
        embeddings.append(np.load(embedding_paths[i]))
    return embeddings


def load_data(data_path):
    data = hdf5storage.read(path='/data', filename=data_path)
    return data


def generate_trial_mask(data, trial_query):
    """Generate a mask for trials based on the given query over the data."""
    return eval(trial_query).flatten()
