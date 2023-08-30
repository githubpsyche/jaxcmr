__all__ = ['load_parameters', 'load_embeddings', 'load_data', 'generate_trial_mask']

import numpy as np
import hdf5storage
import json


def load_parameters(parameter_path: str) -> dict:
    """Load parameters from json file."""
    with open(parameter_path, 'r') as f:
        parameters = json.load(f)
    return parameters


def load_embeddings(embedding_paths) -> list[np.ndarray]:
    """Load embeddings from npy files."""
    return [np.load(embedding_paths[i]) for i in range(len(embedding_paths))]


def load_data(data_path) -> dict:
    """Load data from hdf5 file."""
    return hdf5storage.read(path='/data', filename=data_path)


def generate_trial_mask(data, trial_query) -> np.ndarray:
    """Generate a mask for trials based on the given query over the data."""
    return eval(trial_query).flatten()
