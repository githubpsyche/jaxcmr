"""Input setup must preserve local work and fetch only the requested examples."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jaxcmr import colab


REVISION = "a" * 40


@pytest.mark.parametrize("revision", [REVISION, "main"])
def test_missing_input_is_verified_and_reused(tmp_path, monkeypatch, revision):
    monkeypatch.chdir(tmp_path)
    path = Path("data/example.txt")
    content = b"example input\n"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(colab, "_INPUT_HASHES", {str(path): digest})
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        path.parent.mkdir()
        path.write_bytes(content)

    monkeypatch.setitem(sys.modules, "pooch", SimpleNamespace(retrieve=retrieve))
    colab.fetch_inputs([str(path)], revision=revision)
    colab.fetch_inputs([path], revision=revision)
    assert calls == [{
        "url": f"https://raw.githubusercontent.com/githubpsyche/jaxcmr/{revision}/data/example.txt",
        "known_hash": f"sha256:{digest}", "fname": "example.txt", "path": Path("data"),
    }]
    assert path.read_bytes() == content


def test_conflicting_input_is_reported_and_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("example.txt")
    path.write_bytes(b"local work")
    monkeypatch.setattr(colab, "_INPUT_HASHES", {str(path): "0" * 64})
    with pytest.raises(ValueError, match="Preserved conflicting input example.txt"):
        colab.fetch_inputs([path], revision=REVISION)
    assert path.read_bytes() == b"local work"


def test_custom_paths_are_left_user_managed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("custom.h5")
    path.write_bytes(b"custom observations")
    colab.fetch_inputs([path, "missing/custom.h5"], revision=REVISION)
    assert path.read_bytes() == b"custom observations"
    assert list(tmp_path.iterdir()) == [tmp_path / path]
