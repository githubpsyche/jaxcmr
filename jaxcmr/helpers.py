"""Read and save array files, resolve project paths, and support configured calls.

``load_hdf5_arrays`` reads named arrays from JAXCMR's HDF5 layout; ``load_data``
uses it to load recall datasets and optionally limit participants.
``save_dict_to_hdf5`` writes arrays in the same layout. Use the generic reader
for observations that do not have the fields of a recall dataset.

``find_project_root`` searches for an enclosing project and fails when none is
found. ``resolve_project_root`` instead chooses a working directory, using an
explicit path, Colab's current directory, or Git discovery with a
current-directory fallback. ``resolve_project_path`` resolves filenames against
an explicit directory. All three return ``pathlib.Path`` objects.

``import_from_string`` resolves dotted Python names for configured calls, and
``format_floats`` formats numeric labels. Figure saving lives in
``jaxcmr.plotting``; dataset operations live in ``jaxcmr.dataset``.
"""

import importlib
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np

from jaxcmr.dataset import limit_to_first_subjects
from jaxcmr.typing import Array, RecallDataset, Shaped

__all__ = [
    "find_project_root",
    "format_floats",
    "import_from_string",
    "load_data",
    "load_hdf5_arrays",
    "resolve_project_path",
    "resolve_project_root",
    "save_dict_to_hdf5",
]


def find_project_root(marker: str = ".git") -> Path:
    """Find the nearest enclosing directory containing a marker.

    Parameters
    ----------
    marker : str, optional
        File or directory name to seek, starting in the current directory.
        Defaults to .git; a Git worktree's marker file also qualifies.

    Returns
    -------
    pathlib.Path
        Absolute path to the first matching directory.

    Raises
    ------
    FileNotFoundError
        No matching directory exists between the current directory and its root.
    """
    working_directory = Path.cwd()
    for directory in [working_directory, *working_directory.parents]:
        if (directory / marker).exists():
            return directory
    raise FileNotFoundError(f"Could not find project root containing {marker}.")


def resolve_project_path(path: str | Path, base_dir: str | Path) -> Path:
    """Resolve a filename against an explicit base directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Native filesystem path. Relative paths are joined to base_dir;
        absolute paths retain their own root. A leading tilde is expanded.
    base_dir : str or pathlib.Path
        Base for relative paths. A relative base is itself interpreted against
        the current directory.

    Returns
    -------
    pathlib.Path
        Absolute resolved path, with parent references and symlinks resolved.
        The destination need not exist; no file or directory is created.
    """
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(base_dir).expanduser() / path).resolve()


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Choose the working project directory, with a current-directory fallback.

    Parameters
    ----------
    project_root : str, pathlib.Path or None, optional
        Explicit directory, resolved against the current directory. When omitted,
        Colab uses the current directory; elsewhere the nearest enclosing Git
        project is used, falling back to the current directory if none is found.

    Returns
    -------
    pathlib.Path
        Absolute project directory. The function does not print, change the
        working directory, create directories, or fetch inputs.
    """
    if project_root is not None:
        return resolve_project_path(project_root, Path.cwd())
    if "google.colab" in sys.modules:
        return Path.cwd()
    try:
        return find_project_root()
    except FileNotFoundError:
        return Path.cwd()


def load_hdf5_arrays(path: str | Path) -> dict[str, Array]:
    """Read named arrays from the transposed /data HDF5 layout.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file whose /data group contains array datasets.

    Returns
    -------
    dict[str, Array]
        Each stored array is transposed with .T and converted to a JAX array.
        For matrices, this restores trial rows. No recall fields are required,
        and no participant selection is performed. JAX's configured precision
        determines whether 64-bit values retain their storage dtype.

    Raises
    ------
    KeyError
        The /data group is absent.
    TypeError
        The /data object is not an HDF5 group.
    """
    with h5py.File(path, "r") as data_file:
        data_group = data_file["/data"]
        if not isinstance(data_group, h5py.Group):
            raise TypeError("Expected /data to be an HDF5 group.")
        return {
            name: jnp.array(values[()].T)
            for name, values in data_group.items()
        }


def load_data(data_path: str | Path, max_subjects: int = 0) -> RecallDataset:
    """Load a recall dataset, optionally keeping the lowest participant IDs.

    Parameters
    ----------
    data_path : str or pathlib.Path
        HDF5 file in the layout read by load_hdf5_arrays. It must supply subject,
        listLength, pres_itemnos and recalls with the shapes and dtypes described
        by RecallDataset. Additional array fields are retained.
    max_subjects : int, optional
        Nonnegative participant limit, ordered by ID. Zero keeps everyone.

    Returns
    -------
    RecallDataset
        Recall data with stored arrays transposed into trial rows and converted
        to JAX arrays. Participant selection applies to every field.

    Raises
    ------
    KeyError
        The /data group or a required recall field is absent.
    TypeError
        The /data object is not an HDF5 group.
    """
    data_arrays = load_hdf5_arrays(data_path)
    if max_subjects != 0:
        data_arrays = limit_to_first_subjects(data_arrays, max_subjects)
    dataset: RecallDataset = {
        "subject": data_arrays["subject"],
        "listLength": data_arrays["listLength"],
        "pres_itemnos": data_arrays["pres_itemnos"],
        "recalls": data_arrays["recalls"],
    }
    for name, values in data_arrays.items():
        dataset[name] = values
    return dataset


def save_dict_to_hdf5(
    data: Mapping[str, Shaped[Array | np.ndarray, "..."]],
    path: str | Path,
) -> None:
    """Write named arrays in the transposed /data HDF5 layout.

    Parameters
    ----------
    data : mapping of str to NumPy or JAX arrays
        Arrays stored under their supplied names. Each array is transposed with
        .T before writing; load_hdf5_arrays reverses that transformation.
        Recall-specific fields are not required.
    path : str or pathlib.Path
        Destination file. An existing file is overwritten, including its other
        groups. The parent directory must already exist.

    Returns
    -------
    None
        The file is written and closed; input arrays remain unchanged.
    """
    with h5py.File(path, "w") as data_file:
        data_group = data_file.create_group("data")
        for name, values in data.items():
            data_group.create_dataset(name, data=values.T)


def import_from_string(import_string: str) -> Any:
    """Import an attribute named by a dotted module path.

    Parameters
    ----------
    import_string : str
        Module path followed by one attribute name, such as module.function.
        The module is imported normally, including any module initialization.

    Returns
    -------
    Any
        The named Python object. Its interface depends on the requested object;
        the caller determines whether it is suitable for the intended operation.

    Raises
    ------
    ValueError
        The path has no module/attribute separator.
    ImportError
        The module cannot be imported.
    AttributeError
        The module does not contain the named attribute.
    """
    module_name, attribute_name = import_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def format_floats(iterable: Iterable[float], precision: int = 2) -> list[str]:
    """Format numeric values with a common number of decimal places.

    Parameters
    ----------
    iterable : iterable of float
        Values to format, in label order.
    precision : int, optional
        Number of digits after the decimal point. Defaults to two.

    Returns
    -------
    list of str
        Fixed-point labels in input order, using Python's numeric formatting.
    """
    number_format = f"{{:.{precision}f}}"
    return [number_format.format(value) for value in iterable]
