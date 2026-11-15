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


@pytest.mark.parametrize('template', ['cat_recall_rate','cat_spc'])
@pytest.mark.parametrize('mode', ['single_override','one_in_list','two_in_list'])
def test_custom_category_data_selection_layout_and_paired_outputs(tmp_path, monkeypatch, mode, template):
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.delitem(__import__('sys').modules, 'google.colab', raising=False)
    data = dict(
        subject=jnp.repeat(jnp.arange(1,5),2)[:,None],
        listLength=jnp.full((8,1),2), pres_itemnos=jnp.tile(jnp.array([1,2,0]),(8,1)),
        recalls=jnp.array([[1,0,0],[2,0,0],[1,2,0],[1,0,0],[2,0,0],[2,1,0],[1,0,0],[1,2,0]]),
        kind=jnp.tile(jnp.array([2,5,0]),(8,1)), condition=jnp.tile(jnp.array([1,2]),4),
    )
    source=tmp_path/'my-responses.h5'
    save_dict_to_hdf5(data,str(source)); original=source.read_bytes()
    nb=json.loads((ROOT/f'templates/{template}.ipynb').read_text())
    namespace={}
    transform=InteractiveShell.instance().transform_cell
    def unexpected_install(*args): raise AssertionError('Local execution attempted installation')
    namespace['get_ipython']=lambda: type('Shell',(),{'run_line_magic':unexpected_install})()
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        exec(compile(transform(''.join(cell['source'])),'category-template','exec'),namespace)
        if 'parameters' in cell.get('metadata',{}).get('tags',[]):
            namespace.update(project_root=str(tmp_path),data_path=str(source) if mode=='single_override' else None,
                data_paths=['unused-missing.h5'] if mode=='single_override' else [str(source)]*(2 if mode=='two_in_list' else 1),
                data_labels=['Custom','Replica'] if mode=='two_in_list' else ['Custom'],
                output_dir='results',category_field='kind',category_values=[2,5],category_labels=['A','B'],
                trial_queries=["data['condition'] == 1", "data['condition'] == 2"],labels=['First','Second'] if template=='cat_recall_rate' else None,trial_labels=['First','Second'],
                selection_label='Custom',comparisons=[('Within first',0,0,0,1)],
                contrast_of_contrasts=[('Change',(0,0),(0,1),(1,0),(1,1))])
    assert source.read_bytes()==original
    if template == 'cat_spc':
        assert len(namespace['panels']) == (2 if mode == 'single_override' else 4)
        assert namespace['panels'][0]['categories'] == ([2,5] if mode == 'single_override' else [2])
        assert (tmp_path/'results/category_spc.png').stat().st_size > 0
        plt.close('all')
        return
    assert len(namespace['panels'])==(1 if mode=='single_override' else 2)
    rows=list(csv.DictReader((tmp_path/'results/participant_category_rates.csv').open()))
    assert len(rows)==(16 if mode=='two_in_list' else 8)
    first=next(r for r in rows if r['Condition']=='First' and r['Subject']=='1')
    assert float(first['A'])==1 and float(first['B'])==0
    assert set(namespace['cells'][0,0])=={1,2,3,4}
    assert (tmp_path/'results/category_rates.png').stat().st_size>0
    plt.close('all')


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
