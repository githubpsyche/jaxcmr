"""Project-directory discovery and path resolution."""

from pathlib import Path
import sys

import pytest

from jaxcmr.helpers.paths import resolve_project_root


def test_root_precedence_and_no_side_effects(tmp_path,monkeypatch,capsys):
    project=tmp_path/'project'; nested=project/'analysis';nested.mkdir(parents=True)
    (project/'.git').mkdir()
    monkeypatch.chdir(nested);monkeypatch.delitem(sys.modules,'google.colab',raising=False)
    assert resolve_project_root()==project
    explicit=resolve_project_root('../my-output')
    assert explicit==project/'my-output' and not explicit.exists()
    assert Path.cwd()==nested
    monkeypatch.setitem(sys.modules,'google.colab',object())
    assert resolve_project_root()==nested
    assert resolve_project_root(project)==project
    assert capsys.readouterr().out==''


def test_root_fallback_and_path_validation(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.delitem(sys.modules,'google.colab',raising=False)
    assert resolve_project_root()==tmp_path
    with pytest.raises(ValueError):resolve_project_root('C:relative')
    if sys.platform!='win32':
        with pytest.raises(ValueError):resolve_project_root('C:/foreign/path')
