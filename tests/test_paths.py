"""Project discovery is strict; working-directory selection has a fallback."""

import sys
from pathlib import Path

import pytest

from jaxcmr.helpers import find_project_root, resolve_project_root


def test_root_precedence_and_no_side_effects(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    nested = project / "analysis"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    found = find_project_root()
    assert isinstance(found, Path)
    assert found == project
    assert resolve_project_root() == project
    explicit = resolve_project_root("../my-output")
    assert explicit == project / "my-output" and not explicit.exists()
    assert Path.cwd() == nested
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert resolve_project_root() == nested
    assert resolve_project_root(project) == project
    assert capsys.readouterr().out == ""


def test_root_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    with pytest.raises(FileNotFoundError, match="missing-project-marker"):
        find_project_root("missing-project-marker")
    assert resolve_project_root() == tmp_path
