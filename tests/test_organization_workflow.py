"""Regression checks for explicit rank definitions and notebook artifact options."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap
from jaxcmr.analyses.lagrank import item_midrank_trial, lagrank, tabulate_trial
from jaxcmr.analyses.uncertainty import mean_interval
from jaxcmr.helpers import save_figure


def reference(sequence, length):
    seen, scores = set(), []
    for previous, current in zip(sequence, sequence[1:]):
        if previous <= 0:
            continue
        seen.add(previous)
        if current <= 0 or current in seen:
            continue
        pool = np.array([abs(item - previous) for item in range(1, length + 1) if item not in seen])
        if len(pool) > 1:
            target = abs(current - previous)
            scores.append((len(pool)-1-np.sum(pool < target)-0.5*(np.sum(pool == target)-1))/(len(pool)-1))
    return np.mean(scores) if scores else np.nan


def test_midrank_retains_tied_items_and_excludes_no_choice():
    np.testing.assert_allclose(item_midrank_trial(jnp.array([2, 1, 0]), jnp.arange(1, 4)), 0.5)
    assert np.isnan(item_midrank_trial(jnp.array([1, 2]), jnp.arange(1, 3)))
    assert np.isnan(item_midrank_trial(jnp.array([0, 0]), jnp.arange(1, 3)))
    assert np.isnan(item_midrank_trial(jnp.array([2, 0]), jnp.arange(1, 3)))
    rng = np.random.default_rng(71)
    sequences = []
    for _ in range(40):
        n = int(rng.integers(2, 8))
        sequences.append(np.pad(rng.permutation(np.arange(1, 8))[:n], (0, 7-n)))
    data = dict(recalls=jnp.array(sequences), pres_itemnos=jnp.tile(jnp.arange(1, 8), (40, 1)))
    result = jax.jit(lagrank, static_argnames=('rank_method',))(data, rank_method='item_midrank')
    np.testing.assert_allclose(result, [reference(s, 7) for s in sequences], atol=1e-7, equal_nan=True)
    old = jax.vmap(tabulate_trial, in_axes=(0, 0, None))(data['recalls'], data['pres_itemnos'], 3)
    np.testing.assert_allclose(lagrank(data), old, equal_nan=True)


def test_midrank_does_not_claim_repeated_study_support():
    assert np.isnan(item_midrank_trial(jnp.array([1, 3, 0]), jnp.array([1, 1, 2])))


def test_bootstrap_configuration_is_explicit_and_reproducible():
    values = np.array([0.11, 0.25, 0.6, 0.9, 0.93])
    result = mean_interval(values, seed=19, n_resamples=399)
    expected = bootstrap((values,), np.mean, method='BCa', n_resamples=399,
                         rng=np.random.default_rng(19)).confidence_interval
    np.testing.assert_allclose([result['low'], result['high']], expected)
    assert result == mean_interval(values, seed=19, n_resamples=399)
    assert mean_interval([0.5, 0.5])['low'] == 0.5


def test_figure_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.2, 0.5])
    save_figure(str(tmp_path), 'plot', formats=['png', 'svg'])
    assert (tmp_path/'plot.png').stat().st_size > 0
    assert '<svg' in (tmp_path/'plot.svg').read_text()
    plt.close(fig)


def test_simultaneous_intervention_preserves_baseline(tmp_path):
    import hashlib
    from types import SimpleNamespace
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
    namespace = dict(np=np, jnp=jnp, jax=jax, json=json, hashlib=hashlib,
        parameter_sweeps={}, parameter_interventions={'off': {'fit': 'm', 'overrides': {'s_emotional': 0., 's_neutral': 0.}}},
        records={'m': record}, selections={'m': jnp.ones(2, bool)}, data={'subject': jnp.array([[1], [2]])},
        figure_dir=None, output_dir=tmp_path, construct_model=lambda _: None, load_features=lambda _: None,
        import_from_string=lambda _: None, simulate_h5_from_h5=simulate,
        save_dict_to_hdf5=lambda data, path: Path(path).touch(), fit_paths={'m': fit_path},
        simulation_seed=1, experiment_count=2, data_sha256='fixture', enable_x64=True,
        jaxcmr=SimpleNamespace(__version__='test'), provenance={})
    for cell in nb['cells']:
        code = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and ('7 — Prepare sweeps' in code or '8 — Simulate and save' in code):
            exec(compile(code, str(template), 'exec'), namespace)
    assert len(seen) == 2
    np.testing.assert_array_equal(
        seen[0]['s_emotional'], np.full(2, 1.1, dtype=seen[0]['s_emotional'].dtype)
    )
    np.testing.assert_array_equal(seen[1]['s_emotional'], [0, 0])
    np.testing.assert_array_equal(seen[1]['s_neutral'], [0, 0])
    np.testing.assert_array_equal(seen[0]['other'], seen[1]['other'])
    assert record['fits'] == parameters
