import importlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import h5py
import jax.numpy as jnp
import markdown
from IPython.display import HTML, display
from jax import vmap
from nbdev import show_doc as nbdev_show_doc

from jaxcmr.typing import Array, Bool, Float, Real, Integer


def show_doc(func):
    """Decorator to show documentation for a function."""
    html = markdown.markdown(str(nbdev_show_doc(func)), extensions=["tables"])
    display(HTML(html))


def all_rows_identical(arr: Real[Array, " x y"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0])  # type: ignore


def log_likelihood(likelihoods: Float[Array, "trial_count ..."]) -> Float[Array, ""]:
    """Return the summed log likelihood over specified likelihoods."""
    return -jnp.sum(jnp.log(likelihoods))


def import_from_string(import_string):
    """
    Import a module or function from a string.

    Args:
        import_string: A string in the format 'module.submodule.ClassName' or 'module.function_name'.

    Returns:
        The imported module or function.

    Raises:
        ImportError: If the import string is not valid.
    """
    module_name, function_name = import_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def format_floats(iterable: Iterable[float], precision: int = 2) -> List[str]:
    """
    Formats a list of floats to a specified precision.

    Args:
        iterable: Iterable of floats to format.
        precision: Number of decimal places to format to.

    Returns:
        List of formatted strings.
    """
    format_str = f"{{:.{precision}f}}"
    return [format_str.format(x) for x in iterable]


def find_project_root(marker: str = ".git") -> str:
    """
    Finds the project root by traversing upwards from `start` directory
    until a directory containing `marker` is found.
    """
    start = Path.cwd()
    for path in [start, *start.parents]:
        if (path / marker).exists():
            return str(path)
    raise FileNotFoundError(f"Could not find project root containing {marker}.")


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


def has_repeats_per_row(arr: Integer[Array, " T L"]) -> Bool[Array, " T"]:
    """
    Returns a boolean array indicating which rows in `recalls` contain repeated nonzero values.

    Args:
        arr: 2D array
    """

    def check_row(row):
        # Only consider nonzero values
        mask = row > 0
        values: Integer[Array, " values"] = jnp.where(
            mask, row, -1
        )  # fill non-recalled with -1 (guaranteed not to repeat)

        # Sort and compare adjacent values
        sorted_vals = jnp.sort(values)
        eq_adjacent = sorted_vals[1:] == sorted_vals[:-1]

        # Must ignore -1/-1 matches (which occur if multiple padding values present)
        valid_repeats = eq_adjacent & (sorted_vals[1:] != -1)
        return jnp.any(valid_repeats)

    return vmap(check_row)(arr)
