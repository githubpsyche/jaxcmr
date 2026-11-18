"""Check source preservation and safe website staging without version machinery."""
import importlib.util
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

from IPython.core.interactiveshell import InteractiveShell
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_prepare", ROOT / "guide/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def staging(tmp_path, monkeypatch):
    guide = tmp_path / "guide"
    guide.mkdir()
    (tmp_path / "templates").mkdir()
    (tmp_path / "data/fits").mkdir(parents=True)
    (guide / "examples").mkdir()
    raw = json.dumps({"cells": [
        {"cell_type": "raw", "source": ["---\n", "title: Example\n", "format:\n", "  html:\n", "    code-links:\n", "---\n"]},
        {"cell_type": "code", "id": "parameters", "metadata": {"tags": ["parameters"]}, "source": ["x = 3"], "outputs": []},
    ]}).encode()
    (tmp_path / "templates/example.ipynb").write_bytes(raw)
    (guide / "developer-setup.ipynb").write_bytes(raw)
    (guide / "_quarto.yml").write_text("  - templates/example.ipynb\n")
    (tmp_path / "data/example.txt").write_bytes(b"example")
    (tmp_path / "data/fits/README.md").write_text("Provenance")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "GUIDE", guide)
    monkeypatch.setattr("jaxcmr.examples._EXAMPLE_FILES", ("data/example.txt",))
    monkeypatch.delenv("QUARTO_PROJECT_RENDER_ALL", raising=False)
    return tmp_path, guide, raw


def test_staging_preserves_download_and_executable_content(staging):
    root, guide, raw = staging
    prepare.stage()
    assert (guide / "downloads/example.ipynb").read_bytes() == raw
    staged = json.loads((guide / "templates/example.ipynb").read_bytes())
    assert staged["cells"][1:] == json.loads(raw)["cells"][1:]
    assert "blob/main/templates/example.ipynb" in ''.join(staged["cells"][0]["source"])
    assert (root / "templates/example.ipynb").read_bytes() == raw
    assert (guide / "templates/data/example.txt").read_bytes() == (root / "data/example.txt").read_bytes()


def test_full_render_cannot_clear_source_directory(staging, monkeypatch):
    root, _, _ = staging
    monkeypatch.setenv("QUARTO_PROJECT_RENDER_ALL", "1")
    monkeypatch.setenv("QUARTO_PROJECT_OUTPUT_DIR", str(root / "templates"))
    with pytest.raises(ValueError, match="source directory"):
        prepare.stage()
    assert (root / "templates/example.ipynb").exists()


def test_notebook_installation_only_runs_in_colab(monkeypatch):
    names = re.findall(r"^\s*- (templates/\w+\.ipynb)\s*$", (ROOT / "guide/_quarto.yml").read_text(), re.M)
    paths = [*[ROOT / name for name in names], ROOT / "guide/examples/first-simulation.ipynb",
             ROOT / "guide/examples/war-photo-data-preparation.ipynb", ROOT / "guide/tutorials/data-preparation.ipynb"]
    shell = InteractiveShell.instance()
    for path in paths:
        nb = json.loads(path.read_bytes())
        cells = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "%pip install" in ''.join(c["source"])]
        assert len(cells) == 1, path
        calls = []
        fake_shell = SimpleNamespace(run_line_magic=lambda *args: calls.append(args))
        source = shell.transform_cell(cells[0])
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        exec(source, {"get_ipython": lambda: fake_shell})
        assert not calls, path
        monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace())
        exec(source, {"get_ipython": lambda: fake_shell})
        assert len(calls) == 1 and calls[0][0] == "pip", path
