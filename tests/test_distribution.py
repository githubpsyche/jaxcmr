"""Distribution contents, ordinary pip installation, and bundled example resources."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from jaxcmr.examples import _EXAMPLE_FILES
import pytest

SOURCE = Path(os.environ.get("JAXCMR_SOURCE", Path(__file__).resolve().parents[1]))
VERSION = tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def run(*args, cwd, env=None):
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    work = tmp_path_factory.mktemp("distribution")
    run(
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--outdir",
        str(work / "dist"),
        cwd=SOURCE,
    )
    return work, next((work / "dist").glob("*.whl")), next((work / "dist").glob("*.tar.gz"))


@pytest.fixture(scope="module")
def installed(artifacts):
    work, wheel, sdist = artifacts
    environment = work / "empty-environment"
    run(sys.executable, "-m", "venv", str(environment), cwd=work)
    python = str(environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    # Build from the source archive, so checkout-only files cannot rescue installation.
    run(
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        str(sdist),
        "--wheel-dir",
        str(work / "rebuilt"),
        cwd=work,
        env=env,
    )
    rebuilt = next((work / "rebuilt").glob("*.whl"))
    run(python, "-m", "pip", "install", str(rebuilt), cwd=work, env=env)
    run(python, "-m", "pip", "check", cwd=work, env=env)
    return work, python, env


def test_archives_include_licenses_and_exclude_research(artifacts):
    _, wheel, sdist = artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(name.startswith(("work/", "data/", "results/", "tests/")) for name in names)
        examples = {f"jaxcmr/_example_inputs/{name}" for name in [*_EXAMPLE_FILES, "data/fits/README.md"]}
        assert {name for name in names if name.startswith("jaxcmr/_example_inputs/")} == examples
        for name in [*_EXAMPLE_FILES, "data/fits/README.md"]:
            assert archive.read(f"jaxcmr/_example_inputs/{name}") == (SOURCE / name).read_bytes()
        license_text = archive.read("jaxcmr/_vendor/simple_pytree/LICENSE").decode()
        assert license_text == (SOURCE / "jaxcmr/_vendor/simple_pytree/LICENSE").read_text()
        assert "MIT License" in license_text and "Cristian Garcia" in license_text
        provenance = archive.read("jaxcmr/_vendor/simple_pytree/README.md").decode()
        assert "0.2.2" in provenance and "Cristian Garcia" in provenance
        metadata = archive.read(
            next(n for n in names if n.endswith(".dist-info/METADATA"))
        ).decode()
        assert f"Version: {VERSION}\n" in metadata
        assert not re.search(r"Requires-Dist: simple[-_]pytree", metadata, flags=re.I)
        license_name = next(n for n in names if ".dist-info/" in n and n.endswith("LICENSE"))
        assert b"Apache License" in archive.read(license_name)
    with tarfile.open(sdist) as archive:
        names = {
            Path(*Path(member.name).parts[1:]).as_posix()
            for member in archive.getmembers()
            if member.isfile()
        }
        assert not any(
            name.split("/")[0] in {"work", "results", ".venv", ".git", "_site"}
            for name in names
        )
        assert {name for name in names if name.startswith("data/")} == {*_EXAMPLE_FILES, "data/fits/README.md"}
        for name in [*_EXAMPLE_FILES, "data/fits/README.md"]:
            member = next(m for m in archive.getmembers()
                          if Path(*Path(m.name).parts[1:]).as_posix() == name)
            assert archive.extractfile(member).read() == (SOURCE / name).read_bytes()
        assert {
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "tests/fixtures/recall.json",
            "templates/models/cmr.ipynb",
            "templates/guide/getting-started.qmd",
        } <= names


def test_ordinary_pip_install_and_actual_readme_example(installed):
    work, python, env = installed
    (block,) = re.findall(r"```python\n(.*?)```", (SOURCE / "README.md").read_text(encoding="utf-8"), flags=re.S)
    checks = """
import importlib.metadata as metadata
import importlib.util
import numpy as np
assert importlib.util.find_spec('simple_pytree') is None
assert not any(d.metadata['Name'].replace('_', '-').lower() == 'simple-pytree' for d in metadata.distributions())
assert probs.shape == (17,)
assert np.isfinite(probs).all()
np.testing.assert_allclose(probs.sum(), 1., atol=1e-6)
assert int(model.study_index) == 16
"""
    checks += "\nimport jaxcmr\nassert jaxcmr.__version__ == metadata.version('jaxcmr') == " + repr(
        VERSION
    )
    run(python, "-c", block + checks, cwd=work, env=env)


def test_installed_examples_are_available_offline(installed):
    work, python, env = installed
    script = """
from pathlib import Path
import socket
import sys
from jaxcmr.examples import fetch_inputs, _EXAMPLE_FILES
from importlib.metadata import distribution

def no_network(*args, **kwargs):
    raise AssertionError("Bundled examples must work offline")

socket.socket.connect = no_network
project = Path('offline-project')
fetch_inputs(_EXAMPLE_FILES, base_dir=project)
from jaxcmr.colab import fetch_inputs as legacy_fetch_inputs
legacy_fetch_inputs([next(iter(_EXAMPLE_FILES))], base_dir=project)
for name in _EXAMPLE_FILES:
    bundled = distribution("jaxcmr").locate_file(f"jaxcmr/_example_inputs/{name}")
    assert (project / name).read_bytes() == bundled.read_bytes()
"""
    run(python, "-c", script, cwd=work, env=env)
