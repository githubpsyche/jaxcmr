"""Template adaptations and safeguards beyond the default notebook runs."""
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.core.interactiveshell import InteractiveShell

from jaxcmr.helpers import save_dict_to_hdf5

ROOT=Path(__file__).parents[1]


def run_analysis(template, settings):
    notebook = json.loads((ROOT / f"templates/{template}.ipynb").read_bytes())
    namespace = {}
    transform = InteractiveShell.instance().transform_cell
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(transform("".join(cell["source"])), template, "exec"), namespace)
            if "parameters" in cell["metadata"].get("tags", []):
                namespace.update(settings)
    return namespace


@pytest.fixture
def response_settings(tmp_path):
    study = np.tile([1, 2, 3, 4], (8, 1))
    responses = np.array([
        [1, 2, 0, 0], [1, 4, 2, 0], [2, 4, 3, 0], [2, 1, 4, 3],
        [4, 2, 0, 0], [2, 3, 1, 0], [1, 3, 2, 4], [4, 1, 2, 0],
    ])
    path = tmp_path / "responses.h5"
    save_dict_to_hdf5(dict(
        subject=np.repeat([1, 2, 3, 4], 2)[:, None], listLength=np.full((8, 1), 4),
        pres_itemnos=study, pres_itemids=study, recalls=responses,
        kind=np.tile([2, 5, 2, 5], (8, 1)), condition=np.tile([1, 2], 4),
    ), path)
    return dict(project_root=str(tmp_path), data_paths=[str(path)], data_labels=["Observed"],
                trial_queries=["data['condition'] == 1", "data['condition'] == 2"],
                labels=["First", "Second"], output_dir="results", figure_str="analysis",
                category_field="kind", category_values=[2, 5], category_labels=["A", "B"])


@pytest.mark.parametrize("template,mode,figure_count", [
    ("cat_spc", "categories", 2), ("cat_spc", "conditions", 2), ("cat_spc", "datasets", 4),
    ("cat_recall_rate", "conditions", 1), ("cat_recall_rate", "datasets", 2),
])
def test_custom_category_outputs(tmp_path, monkeypatch, response_settings, template, mode, figure_count):
    settings = response_settings | {"comparison": mode}
    if mode == "datasets":
        settings.update(data_paths=settings["data_paths"] * 2, data_labels=["Observed", "Replica"])
    if template == "cat_recall_rate":
        settings.update(category_values=[2, 5, 9], category_labels=["A", "B", "Absent"],
                        comparisons=[("Within first", 0, 0, 0, 1)],
                        contrast_of_contrasts=[("Change", (0, 0), (0, 1), (1, 0), (1, 1))])
    seen = []
    monkeypatch.setattr(plt, "show", lambda: seen.append(plt.gca()))
    path = Path(settings["data_paths"][0])
    before = path.read_bytes()
    namespace = run_analysis(template, settings)
    assert path.read_bytes() == before
    assert len(seen) == figure_count
    output = tmp_path / "results"
    files = sorted(p.name for p in output.glob("*.png"))
    assert len(files) == figure_count
    if figure_count == 1:
        assert files == ["analysis.png"]
    else:
        assert all(name.startswith(f"analysis_{i}_") for i, name in enumerate(files, 1))
    for axis in seen:
        axis.figure.canvas.draw()
        legend = axis.get_legend()
        assert legend is not None
        bounds = legend.get_window_extent()
        assert bounds.x0 >= axis.bbox.x1
        assert axis.figure.bbox.contains(*bounds.p1)
    if template == "cat_spc":
        labels = seen[0].get_legend_handles_labels()[1]
        assert labels == {"categories": ["A", "B"], "conditions": ["First", "Second"],
                          "datasets": ["Observed", "Replica"]}[mode]
    else:
        with (output / "participant_category_rates.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8 * len(settings["data_paths"])
        first = next(r for r in rows if r["Condition"] == "First" and r["Subject"] == "1")
        assert float(first["A"]) == float(first["B"]) == 0.5
        assert all(np.isnan(float(row["Absent"])) for row in rows)
        assert set(namespace["cells"][0, 0]) == {1, 2, 3, 4}
        assert seen[0].get_legend_handles_labels()[1] == ["A", "B"]


@pytest.mark.parametrize("template", ["lagrank", "distrank"])
def test_rank_figures_draw_exported_intervals_under_both_layouts(tmp_path, monkeypatch, response_settings, template):
    distance_path = tmp_path / "distance.npy"
    np.save(distance_path, np.abs(np.subtract.outer(np.arange(4), np.arange(4))))
    settings = response_settings | dict(distance_matrix_path=str(distance_path), figure_str="",
                                         bootstrap_seed=19, n_resamples=107)
    axes = []
    monkeypatch.setattr(plt, "show", lambda: axes.append(plt.gca()))
    earlier = None
    for mode in ["conditions", "datasets"]:
        axes.clear()
        run_analysis(template, settings | {"comparison": mode})
        with (tmp_path / "results/summary_intervals.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        with (tmp_path / "results/participant_ranks.csv").open() as stream:
            participants = list(csv.DictReader(stream))
        assert len(participants) == 8
        assert [int(row["seed"]) for row in rows] == [19, 20]
        if earlier is not None:
            assert rows == earlier
        earlier = rows
        for index, row in enumerate(rows):
            axis = axes[0 if mode == "conditions" else index]
            points = [line for line in axis.lines if line.get_marker() == "o"]
            group = index if mode == "conditions" else 0
            assert points[group].get_ydata()[0] == pytest.approx(float(row["mean"]))
            np.testing.assert_allclose(axis.collections[group].get_segments()[0][:, 1],
                                       [float(row["low"]), float(row["high"])])
            has_reference = any(line.get_label() == "Uniform-choice reference" for line in axis.lines)
            assert has_reference == (template == "distrank")
    if template == "lagrank":
        axes.clear()
        run_analysis(template, settings | {"comparison": "conditions", "rank_method": "item_midrank"})
        assert any(line.get_label() == "Uniform-choice reference" for line in axes[0].lines)



def test_inspecting_a_missing_fit_never_runs_an_optimizer(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        run_analysis("fitting", {
            "project_root": str(tmp_path),
            "fit_action": "inspect",
            "saved_fit_paths": {"example": "missing.json"},
        })
    assert not (tmp_path / "results").exists()


def test_response_reuse_and_forced_regeneration(tmp_path, response_settings, monkeypatch):
    from jaxcmr import simulation

    data_path = Path(response_settings["data_paths"][0])
    record = json.loads((ROOT / "data/fits/Dupertuys2026_CMR_demo.json").read_text())
    record.update(name="repfr_sub0", data_path=data_path.name, pooled=False,
                  selected_trial_indices=[0, 1])
    record["fits"]["subject"] = [1]
    (tmp_path / "participant.json").write_text(json.dumps(record))
    calls = []

    def simulate(factory, data, features, parameters, trial_mask, **kwargs):
        calls.append((parameters["subject"].tolist(), np.flatnonzero(trial_mask).tolist()))
        return {key: values[trial_mask] for key, values in data.items()}

    monkeypatch.setattr(simulation, "simulate_dataset", simulate)
    settings = dict(project_root=str(tmp_path), fit_paths={"model": "participant.json"},
                    data_path=data_path.name, simulation_dir="responses",
                    sim_alg_path=record["simulation_function"], redo_figures=False,
                    experiment_count=1, simulation_seed=17)
    path = tmp_path / "responses/repfr_sub0.h5"
    for force in (False, False, True):
        if force:
            path.with_suffix(".json").write_text("invalid old metadata")
        result = run_analysis("simulation", settings | {"redo_sims": force})
        assert result["simulation_paths"] == {"model": path}
        assert path.exists()
        saved = json.loads(path.with_suffix(".json").read_text())
        assert saved["seed"] == 17 and saved["selected_trial_indices"] == [0, 1]
        np.testing.assert_array_equal(result["simulations"]["model"]["subject"], [[1], [1]])
    assert calls == [([1], [0, 1]), ([1], [0, 1])]


def test_simultaneous_intervention_preserves_baseline(tmp_path, response_settings, monkeypatch):
    from jaxcmr import simulation

    data_path = Path(response_settings["data_paths"][0])
    record = json.loads((ROOT / "data/fits/Dupertuys2026_CMR_demo.json").read_text())
    record.update(data_path=data_path.name, selected_trial_indices=list(range(8)))
    record["fits"].update(s_emotional=[1.1], s_neutral=[2.2], other=[0.3])
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(record))
    calls = []

    def simulate(factory, data, features, parameters, trial_mask, **kwargs):
        calls.append((parameters, np.asarray(kwargs["rng"])))
        return {key: values[trial_mask] for key, values in data.items()}

    monkeypatch.setattr(simulation, "simulate_dataset", simulate)
    settings = dict(project_root=str(tmp_path), fit_paths={"m": "fit.json"},
                    data_path=data_path.name, parameter_sweeps={}, redo_figures=False,
                    parameter_interventions={"off": {"fit": "m", "overrides": {
                        "s_emotional": 0., "s_neutral": 0.}}},
                    simulation_seed=7, experiment_count=2)
    result = run_analysis("parameter_shifting", settings)
    original, changed = calls
    for key, value in [("s_emotional", 1.1), ("s_neutral", 2.2)]:
        np.testing.assert_allclose(original[0][key], np.full(4, value))
        np.testing.assert_array_equal(changed[0][key], np.zeros(4))
    np.testing.assert_array_equal(changed[0]["other"], original[0]["other"])
    np.testing.assert_array_equal(original[1], changed[1])
    assert json.loads(fit_path.read_text()) == record
    paths = result["all_shifted_paths"]["off"]
    assert {label: Path(path).name for label, path in paths.items()} == {
        "Fitted baseline": "baseline.h5", "off": "intervention.h5"}
    saved = json.loads(Path(paths["off"]).with_suffix(".json").read_text())
    assert saved["parameter_overrides"] == settings["parameter_interventions"]["off"]["overrides"]
