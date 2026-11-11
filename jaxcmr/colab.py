"""Fetch the supplied notebook inputs; custom datasets remain user-managed."""

import hashlib
from pathlib import Path
import re

from jaxcmr.helpers import resolve_project_path


# Files live in the repository, not the installed package. Update a hash only
# when deliberately replacing that example input and its recorded provenance.
_INPUT_HASHES = {
    "data/Dupertuys2026.h5": "93ac35ad7f17c907ad369438c0899be45d4f6728250eccfe7ae3757d2fe2468a",
    "data/Dupertuys2026_semantic_distances.npy": "124f0b00c947f00318be0b6799c8ad0d9c8d2747a05c92119745fc4b2565c703",
    "data/fits/Dupertuys2026_CMR_demo.json": "c785ecaa8042b8a62af817f5356e244f3b648f3078d824d5db471f3f9d029cc9",
    "data/fits/Dupertuys2026_eCMR_source_only.json": "7b40bc4938317c74552049452feea7e083230f66440d1b0d359addc351ff9aa8",
    "data/fits/Dupertuys2026_eCMR_broad.json": "b21c44c875f2f0d7309ae4408df9397d1d02a649b3b198b3b7cc815e2dc92f53",
    'data/simulations/Dupertuys2026_CMR_demo_best_of_3.h5': '33f1a329973cf2aa668c4440fa46701e1d2b9ddb232859eb86640bcb2302641d',
    'data/simulations/Dupertuys2026_CMR_demo_best_of_3.json': '0338a6c050fbd6d9a3d3adfcc9e7a00cd507273b8844f77b861f588524f89265',
    'data/simulations/Dupertuys2026_eCMR_broad_demo_best_of_3.h5': 'acba329d06fb8683d941e8b65e39e8c9ce24c230139e8715dd558f64c84f199b',
    'data/simulations/Dupertuys2026_eCMR_broad_demo_best_of_3.json': 'ad48c900ba936a8eac83716f500065d823bd86b7fe30786966a99c5c2ab0b328',
    'data/simulations/Dupertuys2026_eCMR_source_only_demo_best_of_3.h5': '4ef28c5c2d856c366ca7fbd8467cd56ee816e489dc1133d4fec7c109ce54bf48',
    'data/simulations/Dupertuys2026_eCMR_source_only_demo_best_of_3.json': 'b89e3fcdcee8a95febbc449e43599e4d7b648da1211b17422863da4299732280',
}


def fetch_inputs(paths, *, revision, base_dir=None):
    """Fetch registered relative example names into ``base_dir`` (default: cwd).

    ``revision`` is the full Git commit hash used to install JAXCMR. Pooch
    verifies each download. Matching local files are reused; conflicting files
    raise ``ValueError`` and are never replaced. Paths outside the registry are
    left alone, so changing a notebook parameter to a custom input does not
    download an example over it. Requires the optional ``jaxcmr[colab]`` extra
    only when a download is needed. This function does not install packages.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full Git commit hash (40 hexadecimal characters).")
    base = resolve_project_path(base_dir, Path.cwd()) if base_dir is not None else Path.cwd()
    for requested in paths:
        requested = Path(requested)
        key = requested.as_posix()
        expected = _INPUT_HASHES.get(key)
        path = resolve_project_path(requested, base) if base_dir is not None else requested
        if expected is None:
            print(f"Custom input is user-managed: {path}")
            continue
        if path.exists() or path.is_symlink():
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Preserved conflicting input {path}: expected SHA-256 {expected}, "
                    f"found {actual}. Move it aside to fetch the example, or use "
                    "a different path for your custom input."
                )
            print(f"Reusing verified input: {path}")
            continue
        import pooch

        pooch.retrieve(
            url=f"https://raw.githubusercontent.com/githubpsyche/jaxcmr/{revision}/{key}",
            known_hash=f"sha256:{expected}",
            fname=path.name,
            path=path.parent,
        )
