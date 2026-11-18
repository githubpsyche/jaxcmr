"""Copy supplied example inputs into a project.

JAXCMR includes observations, item distances, saved fits and simulated responses
used by its templates. ``copy_example_inputs`` copies requested examples from
the installed package, reuses identical files and refuses to overwrite different
contents. It works offline, including with an editable installation.

The relative filenames in the example catalog identify these supplied files.
Other paths are left for the caller to manage; this module does not load data
or check whether custom inputs are ready to use.
"""

from collections.abc import Iterable
from importlib.metadata import distribution
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

from jaxcmr.helpers import resolve_project_path

# These selected repository files are also bundled with the installed package.
_EXAMPLE_FILES = (
    'data/Dupertuys2026.h5',
    'data/Dupertuys2026_semantic_distances.npy',
    'data/fits/Dupertuys2026_CMR_demo.json',
    'data/fits/Dupertuys2026_eCMR_source_only.json',
    'data/fits/Dupertuys2026_eCMR_broad.json',
    'data/simulations/Dupertuys2026_CMR_demo_best_of_3.h5',
    'data/simulations/Dupertuys2026_CMR_demo_best_of_3.json',
    'data/simulations/Dupertuys2026_eCMR_broad_demo_best_of_3.h5',
    'data/simulations/Dupertuys2026_eCMR_broad_demo_best_of_3.json',
    'data/simulations/Dupertuys2026_eCMR_source_only_demo_best_of_3.h5',
    'data/simulations/Dupertuys2026_eCMR_source_only_demo_best_of_3.json',
)


def copy_example_inputs(
    paths: Iterable[str | Path],
    *,
    base_dir: str | Path | None = None,
) -> None:
    """Copy requested examples from the installed package into a project.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Requested input filenames. Only relative filenames listed in the
        example catalog identify supplied inputs. Absolute paths and unlisted
        filenames are skipped, even when they do not exist.
    base_dir : str or pathlib.Path, optional
        Project directory in which to place the examples. Defaults to the
        current working directory. Each recognized filename retains its
        subdirectories, such as ``data/fits/``.

    Returns
    -------
    None
        Creates missing files and parent directories, and prints whether each
        input was copied, reused or left to the caller.

    Raises
    ------
    ValueError
        A destination for a recognized example contains different bytes.
        The existing file is preserved.
    OSError
        A packaged resource cannot be read or a destination cannot be written.

    Notes
    -----
    Identical existing files are reused without rewriting them. Files are
    processed individually: if a later input fails, earlier copies remain.
    Custom inputs are neither copied nor checked. No network access is needed.

    Examples
    --------
    >>> from jaxcmr.examples import copy_example_inputs
    >>> copy_example_inputs(["data/Dupertuys2026.h5"], base_dir="my-project")
    """
    base_path = (
        resolve_project_path(base_dir, Path.cwd())
        if base_dir is not None else Path.cwd()
    )
    for requested_path in paths:
        requested_path = Path(requested_path)
        relative_path = requested_path.as_posix()
        destination_path = (
            resolve_project_path(requested_path, base_path)
            if base_dir is not None else requested_path
        )
        if relative_path not in _EXAMPLE_FILES:
            print(f"Custom input is user-managed: {destination_path}")
            continue
        # Hatch editable installs keep these resources in the installed wheel.
        bundled_path = distribution("jaxcmr").locate_file(
            f"jaxcmr/_example_inputs/{relative_path}"
        )
        content = cast(Traversable, bundled_path).read_bytes()
        if destination_path.exists() or destination_path.is_symlink():
            if destination_path.read_bytes() != content:
                raise ValueError(
                    f"Preserved conflicting input {destination_path}: its contents differ from "
                    "the supplied example. Move it aside to prepare the example, "
                    "or use a different path for your custom input."
                )
            print(f"Reusing matching input: {destination_path}")
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("xb") as stream:
            stream.write(content)
        print(f"Prepared bundled input: {destination_path}")
