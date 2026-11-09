from types import SimpleNamespace
import jax
import numpy as np
import pytest
from jax import numpy as jnp
import jaxcmr.fitting.scipy as scipy_fitting
from jaxcmr.fitting import EvosaxDE, MapEvosaxDE, ScanEvosaxDE, ScipyDE


@pytest.mark.parametrize("algorithm", [ScipyDE, EvosaxDE, ScanEvosaxDE, MapEvosaxDE])
def test_fitters_return_the_analytic_minimum_for_each_subject(algorithm):
    data = {"subject": jnp.array([[11], [5], [11], [5]])}
    arguments = dict(
        dataset=data,
        features=None,
        base_params={"a": 99.0, "b": 99.0},
        model_factory_cls=DummyModelFactory,
        loss_fn_cls=QuadraticLoss,
        hyperparams={
            "bounds": {"a": [0.0, 1.0], "b": [0.0, 1.0]},
            "num_steps": 80,
            "pop_size": 16,
            "best_of": 1,
            "seed": 19,
            "polish": False,
            "progress_bar": False,
        },
    )
    result = algorithm(**arguments).fit_subjects(jnp.ones(4, dtype=bool))
    repeated = algorithm(**arguments).fit_subjects(jnp.ones(4, dtype=bool))
    np.testing.assert_array_equal(repeated["fitness"], result["fitness"])
    np.testing.assert_array_equal(repeated["fits"]["a"], result["fits"]["a"])
    np.testing.assert_array_equal(repeated["fits"]["b"], result["fits"]["b"])
    assert list(result["fits"]["subject"]) == [5, 11]
    fitted = np.array([result["fits"]["a"], result["fits"]["b"]])
    assert np.all((fitted >= 0) & (fitted <= 1))
    np.testing.assert_allclose(fitted, [[0.25, 0.25], [0.75, 0.75]], atol=1e-3)
    np.testing.assert_allclose(
        result["fitness"], ((fitted - [[0.25], [0.75]]) ** 2).sum(0), atol=1e-7
    )


class DummyModelFactory:
    pass


class QuadraticLoss:
    def __init__(self, model_factory_cls, dataset, features):
        self.target = jnp.asarray([0.25, 0.75])

    def __call__(self, trial_indices, base_params, free_param_names, x):
        target = self.target[: len(list(free_param_names))]
        return jnp.sum(jnp.square(x - target[:, None]), axis=0)


class IndexedNonfiniteLoss:
    def __init__(self, model_factory_cls, dataset, features):
        pass

    def __call__(self, trial_indices, base_params, free_param_names, x):
        values = jnp.square(x[0] - 0.75)
        return values.at[0].set(jnp.nan)


def make_minimal_dataset(subjects):
    return {"subject": np.asarray(subjects).reshape(-1, 1)}


def test_scipyde_passes_explicit_differential_evolution_controls(monkeypatch):
    """``ScipyDE`` forwards SciPy-parity control hyperparameters."""
    calls = []

    def fake_differential_evolution(func, bounds, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(fun=0.0, x=np.array([0.25]), nit=3, success=True)

    monkeypatch.setattr(scipy_fitting, "differential_evolution", fake_differential_evolution)
    fitter = ScipyDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 12,
            "pop_size": 4,
            "relative_tolerance": 0.25,
            "absolute_tolerance": 0.5,
            "cross_over_rate": 0.75,
            "diff_w": (0.5, 1.0),
            "init": "random",
            "polish": False,
            "progress_bar": False,
            "seed": 123,
        },
    )
    result = fitter.fit(np.array([True, True, True]), subject_id=1)
    assert result["nit"] == [3]
    assert result["converged"] == [True]
    assert len(calls) == 1
    assert calls[0]["maxiter"] == 12
    assert calls[0]["popsize"] == 4
    assert calls[0]["tol"] == 0.25
    assert calls[0]["atol"] == 0.5
    assert calls[0]["mutation"] == (0.5, 1.0)
    assert calls[0]["recombination"] == 0.75
    assert calls[0]["init"] == "random"
    assert calls[0]["polish"] is False
    assert isinstance(calls[0]["rng"], np.random.Generator)


def test_evosaxde_explicit_restarts_reproduce_best_of_selection():
    """explicit-key restarts reproduce ``fit(best_of=n)``."""
    hyperparams = {
        "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "num_steps": 10,
        "pop_size": 5,
        "best_of": 3,
        "progress_bar": False,
        "seed": 123,
    }
    trial_mask = np.array([True, True, True])
    combined_fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    restart_fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    combined = combined_fitter.fit(trial_mask, subject_id=1)
    fit_key = jax.random.fold_in(jax.random.PRNGKey(hyperparams["seed"]), 0)
    restart_keys = jax.random.split(fit_key, hyperparams["best_of"])
    restarts = [restart_fitter.fit_once(trial_mask, key, subject_id=1) for key in restart_keys]
    winner = min(enumerate(restarts), key=lambda indexed: (indexed[1]["fitness"][0], indexed[0]))[1]
    assert all((result["hyperparameters"]["best_of"] == 1 for result in restarts))
    assert combined["fitness"] == winner["fitness"]
    assert combined["nit"] == winner["nit"]
    assert combined["converged"] == winner["converged"]
    assert combined["fits"]["subject"] == winner["fits"]["subject"]
    np.testing.assert_allclose(combined["fits"]["x"], winner["fits"]["x"])
    np.testing.assert_allclose(combined["fits"]["y"], winner["fits"]["y"])


def test_evosaxde_resamples_out_of_bounds_candidates_by_default():
    """``EvosaxDE`` repairs out-of-bounds trial values like SciPy."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 4,
            "progress_bar": False,
            "seed": 0,
        },
    )
    population = jnp.asarray([[-1.0, 2.0], [0.25, 0.75]])
    repaired = np.asarray(fitter._ensure_bounds(jax.random.PRNGKey(0), population))
    assert np.all(repaired >= 0.0)
    assert np.all(repaired <= 1.0)
    assert not np.allclose(repaired[0], [0.0, 1.0])
    np.testing.assert_allclose(repaired[1], [0.25, 0.75])


def test_evosaxde_can_clip_out_of_bounds_candidates_for_compatibility():
    """``boundary_handling='clip'`` preserves the old repair behavior."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 4,
            "boundary_handling": "clip",
            "progress_bar": False,
            "seed": 0,
        },
    )
    population = jnp.asarray([[-1.0, 2.0], [0.25, 0.75]])
    repaired = np.asarray(fitter._ensure_bounds(jax.random.PRNGKey(0), population))
    np.testing.assert_allclose(repaired[0], [0.0, 1.0])
    np.testing.assert_allclose(repaired[1], [0.25, 0.75])


@pytest.mark.parametrize("fitter_cls", [EvosaxDE, ScanEvosaxDE, MapEvosaxDE])
def test_evosax_subject_results_override_free_parameter_defaults(fitter_cls):
    """Free defaults must not append extra base values to fitted subject columns."""
    fitter = fitter_cls(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {"x": 0.5, "fixed_value": 2.0},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 2,
            "pop_size": 4,
            "progress_bar": False,
        },
    )
    result = fitter.fit_subjects(jnp.ones(4, dtype=bool))
    assert result["fits"]["subject"] == [1, 2]
    assert result["fits"]["fixed_value"] == [2.0, 2.0]
    assert len(result["fits"]["x"]) == 2
    np.testing.assert_allclose(result["fitness"], (np.asarray(result["fits"]["x"]) - 0.25) ** 2)


def test_scanevosaxde_matches_sequential_evosaxde_on_equal_counts():
    """scanned and sequential EvoSax agree on equal-length subjects."""
    hyperparams = {
        "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "num_steps": 10,
        "pop_size": 5,
        "best_of": 2,
        "progress_bar": False,
        "seed": 0,
    }
    sequential = EvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    scanned = ScanEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    sequential_result = sequential.fit_subjects(np.array([True, True, True, True]))
    batched_result = scanned.fit_subjects(np.array([True, True, True, True]))
    assert batched_result["fits"]["subject"] == sequential_result["fits"]["subject"]
    assert batched_result["nit"] == sequential_result["nit"]
    assert batched_result["converged"] == sequential_result["converged"]
    np.testing.assert_allclose(batched_result["fitness"], sequential_result["fitness"])
    np.testing.assert_allclose(batched_result["fits"]["x"], sequential_result["fits"]["x"])
    np.testing.assert_allclose(batched_result["fits"]["y"], sequential_result["fits"]["y"])


def test_mapevosaxde_matches_sequential_evosaxde_on_equal_counts():
    """mapped and sequential EvoSax agree on equal-length subjects."""
    hyperparams = {
        "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "num_steps": 10,
        "pop_size": 5,
        "best_of": 2,
        "progress_bar": False,
        "seed": 0,
        "map_batch_size": 2,
    }
    sequential = EvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    mapped = MapEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    sequential_result = sequential.fit_subjects(np.array([True, True, True, True]))
    mapped_result = mapped.fit_subjects(np.array([True, True, True, True]))
    assert mapped_result["fits"]["subject"] == sequential_result["fits"]["subject"]
    assert mapped_result["nit"] == sequential_result["nit"]
    assert mapped_result["converged"] == sequential_result["converged"]
    np.testing.assert_allclose(mapped_result["fitness"], sequential_result["fitness"])
    np.testing.assert_allclose(mapped_result["fits"]["x"], sequential_result["fits"]["x"])
    np.testing.assert_allclose(mapped_result["fits"]["y"], sequential_result["fits"]["y"])
    assert mapped_result["hyperparameters"]["map_batch_size"] == 2


def test_scanevosaxde_requires_equal_trial_counts():
    """``ScanEvosaxDE`` raises clearly on unequal selected trial counts."""
    fitter = ScanEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 10,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
        },
    )
    with pytest.raises(ValueError, match="same number of selected trials per subject"):
        fitter.fit_subjects(np.array([True, True, True, True, True]))


def test_evosaxde_replaces_nonfinite_population_fitness():
    """non-finite candidate losses are converted to a finite penalty."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        IndexedNonfiniteLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "nonfinite_penalty": 12345.0,
        },
    )
    population = jnp.ones((fitter.all_hyperparams["population_size"], 1))
    fitness = np.asarray(
        fitter._evaluate_population(fitter._losses.select(jnp.asarray([0])), population)
    )
    assert fitness[0] == 12345.0
    assert np.all(np.isfinite(fitness))


def test_evosaxde_convergence_matches_scipy_formula():
    """convergence uses SciPy's population-energy tolerance rule."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "relative_tolerance": 0.1,
            "absolute_tolerance": 0.0,
        },
    )
    fitness = jnp.asarray([9.0, 10.0, 11.0])
    expected = jnp.std(fitness) <= 0.1 * jnp.abs(jnp.mean(fitness))
    assert bool(fitter._has_converged(fitness)) == bool(expected)


def test_evosaxde_penalized_population_does_not_converge():
    """penalty replacement values do not satisfy convergence."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "relative_tolerance": 1000.0,
            "nonfinite_penalty": 12345.0,
        },
    )
    assert not bool(fitter._has_converged(jnp.asarray([12345.0, 12345.0])))


def test_evosaxde_loose_tolerance_stops_scan_early():
    """loose convergence tolerance skips remaining scan generations."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 10,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "relative_tolerance": 1000.0,
        },
    )
    result = fitter.fit(np.array([True, True, True]), subject_id=1)
    assert result["converged"] == [True]
    assert result["nit"][0] < 10
