"""Check our notebook preparation and result validation."""
import json
from pathlib import Path
import sys

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "guide/tools"))
import check_notebooks as runner
import reader_assertions as checks


def test_preparation_keeps_source_science_and_sets_only_test_copy(tmp_path):
    for item in runner.CASES:
        raw = (runner.ROOT / item["path"]).read_bytes()
        before = json.loads(raw)
        prepared = runner.prepared_case(raw, item, tmp_path)
        for original, copied in zip(before["cells"], prepared.cells[1:-1], strict=True):
            source = original["source"]
            assert copied.source == ("".join(source) if isinstance(source, list) else source)
            if "id" in original:
                assert copied.id == original["id"]
        assert (runner.ROOT / item["path"]).read_bytes() == raw
        assert prepared.metadata.execution.allow_errors is False
    scope = {"data_paths": list(range(4)), "data_labels": list("abcd"), "trial_queries": [1,2,3], "labels": None}
    exec(runner.overrides("analysis", Path("C:/Users/reader's project")), scope)
    assert scope["max_subjects"] == 4 and scope["labels"] is None
    assert scope["data_paths"] == [0,1] and scope["data_labels"] == ["a","b"]


@pytest.mark.parametrize("tag", ["skip-execution", "raises-exception"])
def test_required_sources_cannot_skip_execution(tmp_path, tag):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1", metadata={"tags": [tag]})])
    with pytest.raises(ValueError, match="cannot be skipped"):
        runner.prepared_case(nbformat.writes(notebook).encode(), {"profile":"first", "path":"example.ipynb"}, tmp_path)


def test_corrupt_image_fails_readback(tmp_path):
    folder = tmp_path/"results"; folder.mkdir()
    (folder/"figure.png").write_bytes(b"not an image")
    with pytest.raises(OSError):
        checks.verify_outputs(tmp_path, "analysis")


def test_results_require_passed_cases_and_executed_cells(tmp_path):
    path = tmp_path/"report.xml"
    path.write_text('<testsuites><testsuite><testcase name="one"><system-out>output</system-out></testcase></testsuite></testsuites>')
    assert runner.junit_passed(path, 1)
    assert not runner.junit_passed(path, 2)
    path.write_text('<testsuites><testsuite><testcase name="one"><failure/></testcase></testsuite></testsuites>')
    assert not runner.junit_passed(path, 1)

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1")])
    assert not runner.completed_notebook(notebook)
    notebook.cells[0].execution_count = 1
    assert runner.completed_notebook(notebook)
