"""Required distribution and documentation checks, including ordinary pip resolution."""

import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
import nbformat
from nbclient import NotebookClient
import pytest

SOURCE = Path(os.environ.get("JAXCMR_SOURCE", Path(__file__).resolve().parents[1]))
VERSION = tomllib.loads((SOURCE / "pyproject.toml").read_text())["project"]["version"]


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
    python = str(environment / "bin" / "python")
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
        license_text = archive.read("jaxcmr/_vendor/simple_pytree/LICENSE").decode()
        assert hashlib.sha256(license_text.encode()).hexdigest() == (
            "9c47aa6538111e11a66d673a056439656b9b07e2bb2b1706038994e15c1cf7c7"
        )
        provenance = archive.read("jaxcmr/_vendor/simple_pytree/README.md").decode()
        assert "0.2.2" in provenance and "Cristian Garcia" in provenance
        assert "3a7a2f66883194ab14875dd01c4306c4f102f31d90e7e2d421366f12a2c49bab" in provenance
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
            name.split("/")[0] in {"work", "data", "results", ".venv", ".git", "_site"}
            for name in names
        )
        assert {
            "pyproject.toml",
            "LICENSE",
            "README.md",
            "TESTING.md",
            "testing-audit.json",
            "tests/fixtures/recall.json",
            "templates/models/cmr.ipynb",
            "templates/guide/getting-started.qmd",
        } <= names


def test_ordinary_pip_install_and_actual_readme_example(installed):
    work, python, env = installed
    (block,) = re.findall(r"```python\n(.*?)```", (SOURCE / "README.md").read_text(), flags=re.S)
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


def test_existing_cmr_tutorial_in_fresh_kernel(installed, monkeypatch):
    work, python, env = installed
    run(python, "-m", "pip", "install", "ipykernel", cwd=work, env=env)
    kernel = work / "jupyter" / "kernels" / "jaxcmr-distribution"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "JAXCMR distribution check",
                "language": "python",
                "env": {
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                    "JAX_PLATFORMS": "cpu",
                },
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(work / "jupyter"))
    notebook = nbformat.read(SOURCE / "templates" / "models" / "cmr.ipynb", as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="jaxcmr-distribution",
        resources={"metadata": {"path": str(work)}},
    )
    client.execute()
    cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    assert cells and all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    assert any(output.get("data", {}).get("image/png") for cell in cells for output in cell.outputs)
