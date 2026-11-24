"""Template adaptations and safeguards beyond the default notebook runs."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.core.interactiveshell import InteractiveShell

from jaxcmr.helpers import save_dict_to_hdf5
from jaxcmr.fitting import parameters_for_subjects
from jaxcmr.simulation import load_matching_simulation

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



@pytest.mark.parametrize("template_name,cell_id", [
    ("fitting", "e8348092"), ("simulation", "c48d5c11")
])
def test_response_reuse_and_forced_regeneration(tmp_path, template_name, cell_id):
    template = ROOT / f"templates/{template_name}.ipynb"
    notebook = json.loads(template.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and cell.get("id") == cell_id)
    path = tmp_path / "responses.h5"
    configuration = {"seed": 1, "simulation_function": "example"}
    data = {"subject": jnp.array([[1]]), "listLength": jnp.array([[2]]),
            "pres_itemnos": jnp.array([[1, 2]]), "recalls": jnp.array([[1, 0]])}
    save_dict_to_hdf5(data, path)
    calls = []
    prepared = dict(model_factory=None, features=None, parameters={},
                    trial_mask=jnp.ones(1, bool), configuration=configuration)
    namespace = dict(
        json=json, simulation_inputs={"m": prepared}, simulation_paths={"m": path},
        load_matching_simulation=load_matching_simulation,
        simulate_dataset=lambda *args, **kwargs: calls.append("generate") or data,
        save_dict_to_hdf5=save_dict_to_hdf5, import_from_string=lambda path: None,
        experiment_count=1, simulation_seed=1, filter_repeated_recalls=False, data=data,
        jax=SimpleNamespace(__version__="test", random=SimpleNamespace(PRNGKey=lambda seed: seed)),
        jaxcmr=SimpleNamespace(__version__="test"),
    )
    cases = [(False, set(), False), (True, set(), True)]
    if template_name == "fitting":
        cases.append((False, {"m"}, True))
    for redo_sims, fitted_now, regenerate in cases:
        calls.clear()
        namespace.update(redo_sims=redo_sims, fitted_now=fitted_now)
        # Forced runs must bypass old metadata entirely, even if it is broken.
        path.with_suffix(".json").write_text(
            "not JSON" if regenerate else json.dumps(configuration)
        )
        exec(compile(source, str(template), "exec"), namespace)
        assert calls == (["generate"] if regenerate else [])
        np.testing.assert_array_equal(namespace["simulations"]["m"]["recalls"], data["recalls"])


def test_simultaneous_intervention_preserves_baseline(tmp_path):
    template = Path(__file__).parents[1]/'templates/parameter_shifting.ipynb'
    nb = json.loads(template.read_text())
    parameters = {'subject': [-1], 's_emotional': [1.1], 's_neutral': [2.2], 'other': [0.3]}
    record = dict(fits=parameters, fixed={}, pooled=True, components={}, model_factory='example',
                  simulation_function='example.simulate')
    fit_path = tmp_path/'fit.json'
    fit_path.write_text(json.dumps(record))
    seen = []
    def simulate(factory, data, features, params, mask, **kwargs):
        seen.append({k: np.asarray(v).copy() for k, v in params.items()})
        return {'subject': params['subject'][:, None], 'recalls': jnp.ones((2, 1), int)}
    namespace = dict(np=np, jnp=jnp, jax=jax, json=json,
        parameter_sweeps={}, parameter_interventions={'off': {'fit': 'm', 'overrides': {'s_emotional': 0., 's_neutral': 0.}}},
        parameters_for_subjects=parameters_for_subjects, records={'m': record}, selections={'m': jnp.ones(2, bool)}, data={'subject': jnp.array([[1], [2]])},
        figure_dir=None, output_dir=tmp_path, construct_model=lambda _: None, load_features=lambda _: None,
        import_from_string=lambda _: None, simulate_dataset=simulate,
        save_dict_to_hdf5=lambda data, path: Path(path).touch(), fit_paths={'m': fit_path},
        simulation_seed=1, experiment_count=2, data_source='data/fixture.h5', root_path=tmp_path, enable_x64=True,
        resolve_project_path=lambda path, root: root / path,
        jaxcmr=SimpleNamespace(__version__='test'))
    for cell in nb['cells']:
        code = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and cell.get('id') in {'234747c4', '97354e3a'}:
            exec(compile(code, str(template), 'exec'), namespace)
    assert len(seen) == 2
    np.testing.assert_array_equal(
        seen[0]['s_emotional'], np.full(2, 1.1, dtype=seen[0]['s_emotional'].dtype)
    )
    np.testing.assert_array_equal(seen[1]['s_emotional'], [0, 0])
    np.testing.assert_array_equal(seen[1]['s_neutral'], [0, 0])
    np.testing.assert_array_equal(seen[0]['other'], seen[1]['other'])
    assert record['fits'] == parameters
