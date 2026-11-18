"""Copy bundled examples without overwriting the reader's own files."""

from importlib.metadata import distribution
from pathlib import Path

import pytest

from jaxcmr import examples


def test_bundled_inputs_work_offline_and_reuse_files(tmp_path, monkeypatch):
    # The editable installation must expose the same resources as a regular wheel.
    monkeypatch.chdir(tmp_path)
    examples.copy_example_inputs(examples._EXAMPLE_FILES)
    first = {name: (tmp_path / name).stat().st_mtime_ns for name in examples._EXAMPLE_FILES}
    examples.copy_example_inputs(examples._EXAMPLE_FILES, base_dir=tmp_path)
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
        examples.copy_example_inputs([name], base_dir=tmp_path)
    assert path.read_bytes() == b"my changed observations"


def test_offline_custom_paths_stay_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = Path("own.h5")
    existing.write_bytes(b"my observations")
    examples.copy_example_inputs([existing, "missing.h5", tmp_path / "absolute.h5"])
    assert existing.read_bytes() == b"my observations"
    assert list(tmp_path.iterdir()) == [tmp_path / "own.h5"]
