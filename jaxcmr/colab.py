"""Fetch the supplied notebook inputs; custom datasets remain user-managed."""

import hashlib
from pathlib import Path
import re


# Files live in the repository, not the installed package. Update a hash only
# when deliberately replacing that example input and its recorded provenance.
_INPUT_HASHES = {
    "data/Dupertuys2026.h5": "93ac35ad7f17c907ad369438c0899be45d4f6728250eccfe7ae3757d2fe2468a",
    "data/Dupertuys2026_semantic_distances.npy": "124f0b00c947f00318be0b6799c8ad0d9c8d2747a05c92119745fc4b2565c703",
    "data/fits/Dupertuys2026_CMR_demo.json": "ff06d186b78765e66c261091678497a3f4a7c310331461c960a3bb050e570d78",
    "data/fits/Dupertuys2026_eCMR_source_only.json": "865a2781bafeb9b153c60803ed9fc09efbf58d4e6e14894625be33a7d279f7ed",
    "data/fits/Dupertuys2026_eCMR_broad.json": "4806c8a5e020dac17a716e2b36d8d8791c30baeca8eef5ceaf8f57c708faa50c",
}


def fetch_inputs(paths, *, revision):
    """Fetch recognized example paths into the current working folder.

    ``revision`` is the full Git commit hash used to install JAXCMR. Pooch
    verifies each download. Matching local files are reused; conflicting files
    raise ``ValueError`` and are never replaced. Paths outside the registry are
    left alone, so changing a notebook parameter to a custom input does not
    download an example over it. Requires the optional ``jaxcmr[colab]`` extra
    only when a download is needed. This function does not install packages.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full Git commit hash (40 hexadecimal characters).")
    for requested in paths:
        path = Path(requested)
        expected = _INPUT_HASHES.get(path.as_posix())
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
            url=f"https://raw.githubusercontent.com/githubpsyche/jaxcmr/{revision}/{path.as_posix()}",
            known_hash=f"sha256:{expected}",
            fname=path.name,
            path=path.parent,
        )
