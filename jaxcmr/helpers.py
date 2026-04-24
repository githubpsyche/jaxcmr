"""Data loading and general-purpose helpers.

Provides HDF5 dataset loading, trial and recall mask generation,
subject-level analysis application, and assorted utility functions
used across the package.

"""

import importlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import h5py
import jax.numpy as jnp
import numpy as np  # preserved here in order to evaluate some trial queries
from jax import vmap

from jaxcmr.typing import Array, Bool, Bool_, Float, Integer, Real, RecallDataset


__all__ = [
    "have_common_nonzero",
    "all_rows_identical",
    "log_likelihood",
    "import_from_string",
    "format_floats",
    "find_project_root",
    "generate_trial_mask",
    "generate_recall_mask",
    "make_subject_trial_masks",
    "load_data",
    "limit_to_first_subjects",
    "save_dict_to_hdf5",
    "find_max_list_length",
    "apply_by_subject",
    "has_repeats_per_row",
    "make_dataset",
    "save_figure",
]


def have_common_nonzero(
    a: Integer[Array, " size"],
    b: Integer[Array, " size"],
) -> Bool_:
    """
    Return True iff there exists i, j such that
        a[i] == b[j] > 0   (zeros are treated as padding).

    Works inside `jit` because shapes stay static.
    """
    values_equal = a[:, None] == b[None, :]
    return jnp.any(values_equal & (a > 0)[:, None] & (b > 0)[None, :])


def all_rows_identical(arr: Real[Array, " x y"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0]).item()


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
    data: RecallDataset, trial_query: Optional[str]
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


def generate_recall_mask(
    data: RecallDataset, recall_query: Optional[str]
) -> Bool[Array, " trial_count recall_events"]:
    """Returns a boolean mask for selecting recall events based on a query condition.

    Args:
        data: dict containing trial data arrays, including a "recalls" key with an array.
        recall_query: condition to evaluate, which should return a boolean array aligned
            to recall events. If None, returns a mask that selects all recall events.
    """
    if recall_query is None:
        return jnp.ones_like(data["recalls"], dtype=bool)
    return jnp.asarray(eval(recall_query), dtype=bool)


def make_subject_trial_masks(
    trial_mask: Bool[Array, " trials"], subject_vector: Integer[Array, " trials"]
):
    """Returns a list of subject-specific masks and the list of unique subjects."""
    unique_subjects = np.unique(subject_vector)
    subject_masks = [
        (subject_vector == s) & trial_mask.astype(bool) for s in unique_subjects
    ]
    return subject_masks, unique_subjects


def load_data(data_path: str, max_subjects: int = 0) -> RecallDataset:
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
        result = {key: jnp.array(f["/data"][key][()].T) for key in f["/data"].keys()}  # type: ignore

    if max_subjects == 0:
        return result  # type: ignore
    else:
        return limit_to_first_subjects(result, max_subjects)  # type: ignore


def limit_to_first_subjects(
    data: RecallDataset,
    max_subjects: int,
) -> RecallDataset:
    """Returns dataset restricted to the first `max_subjects` unique subjects.

    Args:
      data: Trial-indexed arrays with a `subject` column shaped [trial_count, 1].
      max_subjects: Maximum number of subjects to retain, preserving encounter order.
    """
    subject_ids = data["subject"].reshape(-1)
    unique_subjects = np.unique(np.asarray(subject_ids))
    cutoff_index = min(max_subjects, unique_subjects.size) - 1
    subject_cutoff = unique_subjects[cutoff_index]
    include_mask = subject_ids <= subject_cutoff
    return {key: value[include_mask] for key, value in data.items()}  # type: ignore


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
    datasets: Sequence[RecallDataset],
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
    data: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    func: Callable,
    *args,
    **kwargs,
) -> list[jnp.ndarray]:
    """Apply ``func`` per subject using the entire masked dataset slice.

    Args:
        data: Dataset containing trial-indexed arrays (e.g., ``recalls``, ``pres_itemids``).
        trial_mask: Boolean mask selecting trials to include.
        func: Callable invoked as ``func(subject_data, *args, **kwargs)`` where
          ``subject_data`` is a masked `RecallDataset`.
        *args: Additional positional arguments forwarded to ``func``.
        **kwargs: Additional keyword arguments forwarded to ``func``.
    """

    subject_indices = data["subject"].flatten()
    results: list[jnp.ndarray] = []

    for subject in jnp.unique(data["subject"]):
        subject_mask = jnp.logical_and(subject_indices == subject, trial_mask)
        if jnp.sum(subject_mask) == 0:
            continue
        subject_dataset: RecallDataset = {
            key: value[subject_mask]  # type: ignore
            for key, value in data.items()
        }
        results.append(func(subject_dataset, *args, **kwargs))

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
        values: Integer[Array, " values"] = jnp.where(mask, row, -1)  # type: ignore # fill non-recalled with -1 (guaranteed not to repeat)

        # Sort and compare adjacent values
        sorted_vals = jnp.sort(values)
        eq_adjacent = sorted_vals[1:] == sorted_vals[:-1]

        # Must ignore -1/-1 matches (which occur if multiple padding values present)
        valid_repeats = eq_adjacent & (sorted_vals[1:] != -1)
        return jnp.any(valid_repeats)

    return vmap(check_row)(arr)


def make_dataset(
    recalls: Integer[Array, "n_trials num_recalled"],
    pres_itemnos: Integer[Array, "n_trials num_presented"] | None = None,
    listLength: Integer[Array, "n_trials 1"] | int | None = None,
    subject: Integer[Array, "n_trials 1"] | int = 1,
    *,
    reps: int = 1,
) -> RecallDataset:
    """Construct a ``RecallDataset`` from flexible inputs.

    Parameters
    ----------
    recalls
        Within-list recall events (1-indexed, 0 = padding). A 1-D vector is
        treated as a single trial.
    pres_itemnos
        Within-list presentation items. When omitted, generated as
        ``arange(1, list_length + 1)`` per trial.
    listLength
        List length per trial. Inferred from *pres_itemnos* when omitted.
        When both are provided, an assertion checks compatibility.
        When neither is provided, inferred as
        ``max(recalls.shape[1], recalls.max())``.
    subject
        Subject identifier per trial. Defaults to ``1`` (single subject).
    reps
        Number of times to tile the resulting trials along axis 0.

    Returns
    -------
    RecallDataset
        Dictionary with keys ``recalls``, ``pres_itemnos``, ``listLength``,
        and ``subject``, all shaped ``(n_trials * reps, ...)``.

    """
    # -- normalise recalls (required) -----------------------------------------
    recalls_arr = jnp.atleast_2d(jnp.asarray(recalls, dtype=jnp.int32))

    # -- normalise pres_itemnos (optional) ------------------------------------
    pres_arr: jnp.ndarray | None = None
    if pres_itemnos is not None:
        pres_arr = jnp.atleast_2d(jnp.asarray(pres_itemnos, dtype=jnp.int32))

    # -- normalise listLength (optional) --------------------------------------
    ll_arr: jnp.ndarray | None = None
    if listLength is not None:
        if isinstance(listLength, int):
            ll_arr = jnp.array([[listLength]], dtype=jnp.int32)
        else:
            ll_arr = jnp.asarray(listLength, dtype=jnp.int32).reshape(-1, 1)

    # -- normalise subject ----------------------------------------------------
    if isinstance(subject, int):
        subj_arr = jnp.array([[subject]], dtype=jnp.int32)
    else:
        subj_arr = jnp.asarray(subject, dtype=jnp.int32).reshape(-1, 1)

    # -- resolve n_trials -----------------------------------------------------
    sizes: list[int] = []
    for arr in (recalls_arr, pres_arr, ll_arr, subj_arr):
        if arr is not None and arr.shape[0] > 1:
            sizes.append(arr.shape[0])
    if sizes:
        n_trials = sizes[0]
        assert all(s == n_trials for s in sizes), (
            f"Multi-trial args disagree on n_trials: {sizes}"
        )
    else:
        n_trials = 1

    # -- tile single-trial args to n_trials -----------------------------------
    if recalls_arr.shape[0] == 1 and n_trials > 1:
        recalls_arr = jnp.tile(recalls_arr, (n_trials, 1))
    if pres_arr is not None and pres_arr.shape[0] == 1 and n_trials > 1:
        pres_arr = jnp.tile(pres_arr, (n_trials, 1))
    if ll_arr is not None and ll_arr.shape[0] == 1 and n_trials > 1:
        ll_arr = jnp.tile(ll_arr, (n_trials, 1))
    if subj_arr.shape[0] == 1 and n_trials > 1:
        subj_arr = jnp.tile(subj_arr, (n_trials, 1))

    # -- infer / validate list_length -----------------------------------------
    if pres_arr is not None and ll_arr is not None:
        assert jnp.all(ll_arr == pres_arr.shape[1]), (
            f"listLength ({ll_arr.ravel()}) incompatible with "
            f"pres_itemnos width ({pres_arr.shape[1]})"
        )
        list_length = int(pres_arr.shape[1])
    elif pres_arr is not None:
        list_length = int(pres_arr.shape[1])
        ll_arr = jnp.full((n_trials, 1), list_length, dtype=jnp.int32)
    elif ll_arr is not None:
        list_length = int(ll_arr[0, 0])
    else:
        list_length = int(max(recalls_arr.shape[1], jnp.max(recalls_arr)))
        ll_arr = jnp.full((n_trials, 1), list_length, dtype=jnp.int32)

    # -- generate default pres_itemnos ----------------------------------------
    if pres_arr is None:
        row = jnp.arange(1, list_length + 1, dtype=jnp.int32)
        pres_arr = jnp.tile(row[None, :], (n_trials, 1))

    # -- apply reps -----------------------------------------------------------
    if reps > 1:
        recalls_arr = jnp.tile(recalls_arr, (reps, 1))
        pres_arr = jnp.tile(pres_arr, (reps, 1))
        ll_arr = jnp.tile(ll_arr, (reps, 1))
        subj_arr = jnp.tile(subj_arr, (reps, 1))

    return {
        "recalls": recalls_arr,
        "pres_itemnos": pres_arr,
        "listLength": ll_arr,
        "subject": subj_arr,
    }  # type: ignore


def save_figure(
    figure_dir: str,
    figure_str: str,
    suffix: Optional[str] = None,
    dpi: int = 600,
) -> None:
    """Save the current matplotlib figure to disk, or just show it.

    Parameters
    ----------
    figure_dir : str
        Directory in which to save the figure.
    figure_str : str
        Base filename (without extension). If empty, the figure is
        shown but not saved.
    suffix : str, optional
        Appended to *figure_str* with an underscore separator.
    dpi : int, optional
        Resolution for the saved image. Default 600.
    """
    import os

    import matplotlib.pyplot as plt

    plt.tight_layout()
    if not figure_str:
        plt.show()
        return
    os.makedirs(figure_dir, exist_ok=True)
    suffix_str = f"_{suffix}" if suffix else ""
    figure_path = os.path.join(figure_dir, f"{figure_str}{suffix_str}.png")
    plt.savefig(figure_path, bbox_inches="tight", dpi=dpi)
    plt.show()
