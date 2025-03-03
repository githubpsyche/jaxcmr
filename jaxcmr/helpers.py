from typing import Callable, Optional, Sequence

import h5py
import jax.numpy as jnp

from jaxcmr.typing import Array, Bool

lb = jnp.finfo(jnp.float32).eps


def generate_trial_mask(
    data: dict, trial_query: Optional[str]
) -> Bool[Array, " trial_count"]:
    """Returns a boolean mask for selecting trials based on a specified query condition.

    Args:
        data: dict containing trial data arrays, including a "recalls" key with an array.
        trial_query: condition to evaluate, which should return a boolean array.
        If None, returns a mask that selects all trials.
    """
    if trial_query is None:
        return jnp.ones(data["recalls"].shape[0], dtype=bool)
    return eval(trial_query).flatten()


def load_data(data_path: str) -> dict[str, jnp.ndarray]:
    """
    Loads and processes an HDF5 dataset from the specified file.

    This function opens the HDF5 file at `data_path`, extracts all datasets stored
    under the "/data" group, transposes each array, and converts them into jnp
    arrays for further processing.

    Args:
        data_path: Path to the HDF5 file containing the dataset.

    Returns:
        A dictionary where each key corresponds to a dataset name and each value is
        a jax.numpy array containing the transposed data.
    """
    with h5py.File(data_path, "r") as f:
        result = {key: f["/data"][key][()].T for key in f["/data"].keys()}  # type: ignore
    return {key: jnp.array(value) for key, value in result.items()}


def save_dict_to_hdf5(data: dict, path: str):
    """Save a dictionary of numpy arrays to an HDF5 file."""
    with h5py.File(path, "w") as file:
        data_group = file.create_group(
            "data"
        )  # Create a group named "data" in the HDF5 file
        for key, value in data.items():
            # Create each dataset within the "data" group
            data_group.create_dataset(key, data=value.T)


def find_max_list_length(
    datasets: Sequence[dict[str, jnp.ndarray]],
    trial_masks: Sequence[Bool[Array, " trial_count"]],
) -> int:
    """Returns highest list length across multiple datasets, given trial masks.

    Args:
        datasets: dataset dicts, each with a key "listLength" mapping to a numpy array.
        trial_masks: Boolean numpy arrays used as masks to filter each dataset.
    """
    return max(
        jnp.max(data["listLength"][trial_mask]).item()
        for data, trial_mask in zip(datasets, trial_masks)
    )


def apply_by_subject(
    data: dict[str, jnp.ndarray],
    trial_mask: Bool[Array, " trial_count"],
    func: Callable,
    *args,
) -> list[jnp.ndarray]:
    """Returns result from applying `func` to each subject's data

    Args:
        data: Dataset containing recalls, presentations, etc., indexed by subject.
        trial_mask: Boolean array specifying which trials to include.
        func: Function to apply per subject. It should accept trials, presentations, list_length, and additional arguments.
        *args: Additional positional arguments to pass to the function.
    """
    trials = data["recalls"]
    presentations = data["pres_itemnos"]
    subject_indices = data["subject"].flatten()
    list_length = jnp.max(data["listLength"][trial_mask]).item()
    results = []

    for subject in jnp.unique(data["subject"]):
        subject_mask = jnp.logical_and(subject_indices == subject, trial_mask)
        if jnp.sum(subject_mask) == 0:
            continue
        results.append(
            func(trials[subject_mask], presentations[subject_mask], list_length, *args)
        )
    return results
