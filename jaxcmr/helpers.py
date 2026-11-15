"""Data loading and general-purpose helpers.

Provides HDF5 dataset loading, trial and recall mask generation,
subject-level analysis application, and assorted utility functions
used across the package.

"""

import importlib
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, Concatenate, Literal

import h5py
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from jaxcmr.dataset import _transform_trial_rows
from jaxcmr.typing import (
    Array,
    Bool,
    Bool_,
    Float,
    Integer,
    Real,
    RecallDataset,
    Shaped,
)

__all__ = [
    "all_rows_identical",
    "apply_by_subject",
    "find_max_list_length",
    "find_project_root",
    "format_floats",
    "generate_recall_mask",
    "generate_trial_mask",
    "has_repeats_per_row",
    "have_common_nonzero",
    "import_from_string",
    "limit_to_first_subjects",
    "load_data",
    "log_likelihood",
    "make_dataset",
    "make_subject_trial_masks",
    "require_pres_itemids",
    "require_valence",
    "resolve_project_path",
    "resolve_project_root",
    "save_dict_to_hdf5",
    "save_figure",
]


def have_common_nonzero(
    a: Integer[Array, " first_items"],
    b: Integer[Array, " second_items"],
) -> Bool_:
    """
    Return True iff there exists i, j such that
        a[i] == b[j] > 0   (zeros are treated as padding).

    Works inside `jit` because shapes stay static.
    """
    values_equal = a[:, None] == b[None, :]
    return jnp.any(values_equal & (a > 0)[:, None] & (b > 0)[None, :])


def all_rows_identical(arr: Real[Array, " rows columns"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0]).item()


def log_likelihood(likelihoods: Float[Array, "trial_count ..."]) -> Float[Array, ""]:
    """Return negative summed log likelihood over the supplied probabilities."""
    return -jnp.sum(jnp.log(likelihoods))


def import_from_string(import_string: str) -> Any:
    """Import the attribute named by a dotted Python path.

    Args:
        import_string: A string in the format 'module.submodule.ClassName' or 'module.function_name'.

    Returns:
        The named attribute, such as a function or class.

    Raises:
        ValueError: If the path has no module/attribute separator.
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no such attribute.
    """
    module_name, attribute_name = import_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def format_floats(iterable: Iterable[float], precision: int = 2) -> list[str]:
    """
    Formats a list of floats to a specified precision.

    Args:
        iterable: Iterable of floats to format.
        precision: Number of decimal places to format to.

    Returns:
        List of formatted strings.
    """
    number_format = f"{{:.{precision}f}}"
    return [number_format.format(value) for value in iterable]


def find_project_root(marker: str = ".git") -> str:
    """
    Finds the project root by traversing upwards from the working directory
    until a directory containing `marker` is found.
    """
    working_directory = Path.cwd()
    for directory in [working_directory, *working_directory.parents]:
        if (directory / marker).exists():
            return str(directory)
    raise FileNotFoundError(f"Could not find project root containing {marker}.")


def resolve_project_path(path: str | Path, base_dir: str | Path) -> Path:
    """Resolve a native path against an explicit base, preserving absolute paths.

    Reject Windows drive-relative paths and paths for another operating system
    instead of silently interpreting them as relative filenames.
    """
    path = Path(path).expanduser()
    windows_path = PureWindowsPath(path)
    if windows_path.drive and not windows_path.is_absolute():
        raise ValueError(f"Use a fully absolute or relative path, not {str(path)!r}.")
    if os.name != "nt" and windows_path.drive:
        raise ValueError(f"Windows path {str(path)!r} must be replaced for this computer.")
    if os.name == "nt":
        if path.root and not path.drive:
            raise ValueError(f"Include the drive in absolute Windows path {str(path)!r}.")
        for component in path.parts[bool(path.anchor):]:
            if component in (".", ".."):
                continue
            if (PureWindowsPath(component).is_reserved() or component.endswith((" ", "."))
                    or any(character in '<>:"|?*' or ord(character) < 32 for character in component)):
                raise ValueError(f"Invalid Windows filename component: {component!r}.")
    return (path if path.is_absolute() else Path(base_dir).expanduser() / path).resolve()


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve an explicit project directory or discover the current project.

    An explicit path is normalized against the current directory with the same
    validation as resolve_project_path. Otherwise Colab uses the current
    directory; elsewhere the nearest enclosing Git root is used, falling back
    to the current directory. Returns a Path without printing, changing the
    working directory, creating directories, or fetching inputs.
    """
    if project_root is not None:
        return resolve_project_path(project_root, Path.cwd())
    if "google.colab" in sys.modules:
        return Path.cwd()
    try:
        return Path(find_project_root())
    except FileNotFoundError:
        return Path.cwd()


def generate_trial_mask(
    data: RecallDataset, trial_query: str | None
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
    data: RecallDataset, recall_query: str | None
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
    trial_mask: Bool[Array, " trials"],
    subject_vector: Integer[Array, " trials"],
) -> tuple[list[Bool[Array, " trials"]], Integer[np.ndarray, " subjects"]]:
    """Returns a list of subject-specific masks and the list of unique subjects."""
    unique_subjects = np.unique(subject_vector)
    subject_masks = [
        (subject_vector == subject_id) & trial_mask for subject_id in unique_subjects
    ]
    return subject_masks, unique_subjects


def load_data(data_path: str | Path, max_subjects: int = 0) -> dict[str, Array]:
    """
    Loads and processes an HDF5 dataset from the specified file.

    This function opens the HDF5 file at `data_path`, extracts all datasets stored
    under the "/data" group, transposes each array, and converts them into jnp
    arrays for further processing.

    Args:
        data_path: Path to the HDF5 file containing the dataset.

    Returns:
        A dictionary where each key corresponds to a dataset name and each value is
        a jax.numpy array containing the transposed data. This storage-level helper
        does not assume a particular experimental dataset schema.

    Raises:
        TypeError: The "/data" object is not an HDF5 group.
    """
    with h5py.File(data_path, "r") as data_file:
        data_group = data_file["/data"]
        if not isinstance(data_group, h5py.Group):
            raise TypeError("Expected /data to be an HDF5 group.")
        data_arrays = {
            name: jnp.array(values[()].T)
            for name, values in data_group.items()
        }

    if max_subjects == 0:
        return data_arrays
    return limit_to_first_subjects(data_arrays, max_subjects)


def limit_to_first_subjects(
    data: Mapping[str, Array],
    max_subjects: int,
) -> dict[str, Array]:
    """Returns dataset restricted to the first `max_subjects` unique subjects.

    Args:
      data: Trial-indexed arrays with a `subject` column shaped [trial_count, 1].
      max_subjects: Maximum number of subjects to retain, ordered by subject ID.
    """
    subject_ids = data["subject"].reshape(-1)
    unique_subjects = np.unique(np.asarray(subject_ids))
    cutoff_index = min(max_subjects, unique_subjects.size) - 1
    subject_cutoff = unique_subjects[cutoff_index]
    include_mask = subject_ids <= subject_cutoff
    return {key: value[include_mask] for key, value in data.items()}


def save_dict_to_hdf5(
    data: Mapping[str, Shaped[Array | np.ndarray, "..."]],
    path: str | Path,
) -> None:
    """Save named NumPy or JAX arrays in the transposed HDF5 data layout."""
    with h5py.File(path, "w") as data_file:
        data_group = data_file.create_group("data")
        for name, values in data.items():
            data_group.create_dataset(name, data=values.T)


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


def require_pres_itemids(
    data: RecallDataset,
) -> Integer[Array, "n_trials num_presented"]:
    """Return cross-list presentation IDs required by semantic analyses."""
    if "pres_itemids" not in data:
        raise ValueError(
            "This operation requires dataset['pres_itemids'] (word-pool item IDs)."
        )
    return data["pres_itemids"]


def require_valence(data: RecallDataset) -> Array:
    """Return per-item valence required by emotional model factories."""
    if "valence" not in data:
        raise ValueError("This operation requires dataset['valence'] (per-item valence).")
    return data["valence"]


def apply_by_subject[Result, **Parameters](
    data: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    func: Callable[Concatenate[RecallDataset, Parameters], Result],
    *args: Parameters.args,
    **kwargs: Parameters.kwargs,
) -> list[Result]:
    """Apply ``func`` per subject using the entire masked dataset slice.

    Args:
        data: Dataset containing trial-indexed arrays (e.g., ``recalls``, ``pres_itemids``).
        trial_mask: Boolean mask selecting trials to include.
        func: Callable invoked as ``func(subject_data, *args, **kwargs)`` where
          ``subject_data`` is a masked `RecallDataset`.
        *args: Additional positional arguments forwarded to ``func``.
        **kwargs: Additional keyword arguments forwarded to ``func``.
    """

    subject_ids = data["subject"].flatten()
    results: list[Result] = []

    for subject_id in jnp.unique(data["subject"]):
        subject_mask = jnp.logical_and(subject_ids == subject_id, trial_mask)
        if jnp.sum(subject_mask) == 0:
            continue
        subject_dataset = _transform_trial_rows(
            data, lambda value: value[subject_mask]
        )
        results.append(func(subject_dataset, *args, **kwargs))

    return results


def has_repeats_per_row(
    arr: Integer[Array, " trials recall_events"],
) -> Bool[Array, " trials"]:
    """
    Returns a boolean array indicating which rows in `recalls` contain repeated nonzero values.

    Args:
        arr: 2D array
    """

    def check_row(row: Integer[Array, " recall_events"]) -> Bool[Array, ""]:
        # Only consider nonzero values
        valid_positions = row > 0
        # Fill non-recalled slots with -1 so they cannot count as repeats.
        padded_values = lax.select(valid_positions, row, jnp.full_like(row, -1))

        # Sort and compare adjacent values
        sorted_values = jnp.sort(padded_values)
        adjacent_matches = sorted_values[1:] == sorted_values[:-1]

        # Must ignore -1/-1 matches (which occur if multiple padding values present)
        valid_repeats = adjacent_matches & (sorted_values[1:] != -1)
        return jnp.any(valid_repeats)

    return vmap(check_row)(arr)


def make_dataset(
    recalls: ArrayLike,
    pres_itemnos: ArrayLike | None = None,
    listLength: ArrayLike | None = None,
    subject: ArrayLike = 1,
    *,
    reps: int = 1,
) -> RecallDataset:
    """Construct the four core trial arrays, broadcasting singleton inputs.

    Args:
        recalls: One- or two-dimensional study-position recall codes, supplied
            as a sequence or array; a vector denotes one trial.
        pres_itemnos: One- or two-dimensional within-list item codes;
            a vector denotes one trial.
            Omit to generate sequential codes, padded to the maximum list length.
        listLength: Scalar, vector, or single-column array of study lengths.
            With presentations, defaults to their storage width. Supply explicit
            lengths for padded lists. Without presentations or lengths, the
            default is max(response width, highest recall code), which cannot
            establish the actual study length of an experiment.
        subject: Scalar, vector, or single-column participant IDs; defaults to 1.
        reps: Positive integer number of copies of the complete trial sequence.

    Returns:
        A new RecallDataset of int32 arrays. Singleton inputs are broadcast to
        the common trial count before repetition. Input arrays are not modified.

    Inputs must be integer-valued and representable as int32, with compatible
    trial counts. Lengths must be nonnegative and agree with presentation
    padding. These are caller responsibilities, not runtime value checks.

    With explicit presentations, construction supports JIT and vmap; ``reps``
    must be static. Generating presentations also works with a static integer
    ``listLength``. Inferring their width from array values requires concrete
    data and must happen before JIT.
    """
    recall_matrix = jnp.atleast_2d(jnp.asarray(recalls, dtype=jnp.int32))
    presentation_matrix, list_lengths = _prepare_study_lists(
        recall_matrix, pres_itemnos, listLength
    )
    subject_ids = jnp.asarray(subject, dtype=jnp.int32).reshape(-1, 1)
    return _broadcast_and_repeat_trials(
        recall_matrix, presentation_matrix, list_lengths, subject_ids, reps
    )


def _prepare_study_lists(
    recalls: Integer[Array, " recall_trials recall_events"],
    presentation_items: ArrayLike | None,
    list_lengths: ArrayLike | None,
) -> tuple[
    Integer[Array, " #study_trials study_events"],
    Integer[Array, " #study_trials 1"],
]:
    """Normalize supplied study lists or generate padded sequential items."""
    if presentation_items is not None:
        presentations = jnp.atleast_2d(
            jnp.asarray(presentation_items, dtype=jnp.int32)
        )
        if list_lengths is None:
            list_lengths = presentations.shape[1]
        length_column = jnp.asarray(list_lengths, dtype=jnp.int32).reshape(-1, 1)
        return presentations, length_column

    if list_lengths is None:
        list_lengths = max(recalls.shape[1], int(recalls.max()))
    length_column = jnp.asarray(list_lengths, dtype=jnp.int32).reshape(-1, 1)
    # Keep a supplied integer static so generation also works under JIT.
    study_width = (
        list_lengths if isinstance(list_lengths, (int, np.integer))
        else int(length_column.max())
    )
    study_positions = jnp.arange(1, study_width + 1, dtype=jnp.int32)[None, :]
    presentations = jnp.where(study_positions <= length_column, study_positions, 0)
    return presentations, length_column


def _broadcast_and_repeat_trials(
    recalls: Integer[Array, " #trials recall_events"],
    presentations: Integer[Array, " #trials study_events"],
    list_lengths: Integer[Array, " #trials 1"],
    subject_ids: Integer[Array, " #trials 1"],
    repetitions: int,
) -> RecallDataset:
    """Align singleton trial rows, then repeat the complete trial sequence."""
    trial_count = max(
        values.shape[0]
        for values in (recalls, presentations, list_lengths, subject_ids)
    )

    def repeat_rows(
        values: Integer[Array, " rows columns"],
    ) -> Integer[Array, " repeated_trials columns"]:
        trial_rows = jnp.broadcast_to(values, (trial_count, values.shape[1]))
        return jnp.tile(trial_rows, (repetitions, 1))

    return {
        "recalls": repeat_rows(recalls),
        "pres_itemnos": repeat_rows(presentations),
        "listLength": repeat_rows(list_lengths),
        "subject": repeat_rows(subject_ids),
    }


def save_figure(
    figure_dir: str | Path,
    figure_str: str,
    suffix: str | None = None,
    dpi: int = 600,
    formats: Sequence[Literal["png", "svg", "pdf"]] = ("png",),
    *,
    figure: Figure | None = None,
) -> None:
    """Save a matplotlib figure to disk, or just show it.

    Parameters
    ----------
    figure_dir : str or pathlib.Path
        Directory in which to save the figure.
    figure_str : str
        Base filename (without extension). If empty, the figure is
        shown but not saved.
    suffix : str, optional
        Appended to *figure_str* with an underscore separator.
    dpi : int, optional
        Resolution for the saved image. Default 600.
    formats : sequence of {"png", "svg", "pdf"}, optional
        Nonempty sequence of output formats. Default ("png",).
    figure : matplotlib.figure.Figure, optional
        Figure to lay out and save; defaults to pyplot's current figure.
        Existing pyplot show behavior is retained. Supplying a figure avoids
        accidentally saving another figure that became current in client code.
    """
    import matplotlib.pyplot as plt

    target_figure = plt.gcf() if figure is None else figure
    target_figure.tight_layout()
    if not figure_str:
        plt.show()
        return
    os.makedirs(figure_dir, exist_ok=True)
    filename_suffix = f"_{suffix}" if suffix else ""
    if not formats:
        raise ValueError("formats must contain at least one output format")
    for image_format in dict.fromkeys(formats):
        figure_path = os.path.join(
            figure_dir, f"{figure_str}{filename_suffix}.{image_format}"
        )
        target_figure.savefig(figure_path, bbox_inches="tight", dpi=dpi)
    plt.show()
