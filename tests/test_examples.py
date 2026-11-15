"""Input setup must preserve local work and fetch only the requested examples."""

from io import BytesIO
from pathlib import Path

import pytest
from jaxcmr import examples


REVISION = "a" * 40


@pytest.mark.parametrize("revision", [REVISION, "main"])
def test_revision_input_is_downloaded_and_compared_on_each_call(tmp_path, monkeypatch, revision):
    monkeypatch.chdir(tmp_path)
    path = Path("data/example.txt")
    content = b"example input\n"
    monkeypatch.setattr(examples, "_EXAMPLE_FILES", (str(path),))
    calls = []

    def retrieve(url, *, timeout):
        calls.append(url)
        return BytesIO(content)

    monkeypatch.setattr(examples, "urlopen", retrieve)
    examples.fetch_inputs([str(path)], revision=revision)
    examples.fetch_inputs([path], revision=revision)
    assert calls == [
        f"https://raw.githubusercontent.com/githubpsyche/jaxcmr/{revision}/data/example.txt"
    ] * 2
    assert path.read_bytes() == content


def test_conflicting_input_is_reported_and_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("example.txt")
    path.write_bytes(b"local work")
    monkeypatch.setattr(examples, "_EXAMPLE_FILES", (str(path),))
    monkeypatch.setattr(examples, "urlopen", lambda *args, **kwargs: BytesIO(b"supplied example"))
    with pytest.raises(ValueError, match="Preserved conflicting input example.txt"):
        examples.fetch_inputs([path], revision=REVISION)
    assert path.read_bytes() == b"local work"


def test_custom_paths_are_left_user_managed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("custom.h5")
    path.write_bytes(b"custom observations")
    examples.fetch_inputs([path, "missing/custom.h5"], revision=REVISION)
    assert path.read_bytes() == b"custom observations"
    assert list(tmp_path.iterdir()) == [tmp_path / path]


def test_bundled_inputs_work_offline_and_reuse_files(tmp_path, monkeypatch):
    # The editable installation must expose the same resources as a regular wheel.
    from importlib.metadata import distribution
    examples.fetch_inputs(examples._EXAMPLE_FILES, base_dir=tmp_path)
    first = {name: (tmp_path / name).stat().st_mtime_ns for name in examples._EXAMPLE_FILES}
    examples.fetch_inputs(examples._EXAMPLE_FILES, base_dir=tmp_path)
    for name in examples._EXAMPLE_FILES:
        path = tmp_path / name
        packaged = distribution("jaxcmr").locate_file(f"jaxcmr/_example_inputs/{name}")
        assert path.read_bytes() == packaged.read_bytes()
        assert path.stat().st_mtime_ns == first[name]


def test_offline_conflict_preserves_local_work(tmp_path):
    name = next(iter(examples._EXAMPLE_FILES))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"my changed observations")
    with pytest.raises(ValueError, match="Preserved conflicting input"):
        examples.fetch_inputs([name], base_dir=tmp_path)
    assert path.read_bytes() == b"my changed observations"


def test_offline_custom_paths_stay_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = Path("own.h5")
    existing.write_bytes(b"my observations")
    examples.fetch_inputs([existing, "missing.h5", tmp_path / "absolute.h5"])
    assert existing.read_bytes() == b"my observations"
    assert list(tmp_path.iterdir()) == [tmp_path / "own.h5"]

