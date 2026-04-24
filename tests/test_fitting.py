import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaxcmr.fitting.scipy as scipy_fitting
import jaxcmr.fitting as fitting
from jaxcmr.fitting import (
    BatchEvosaxDE,
    EvosaxDE,
    MapEvosaxDE,
    ScipyDE,
    ScanEvosaxDE,
)
from jaxcmr.fitting.scipy import make_scipy_loss_fn
from jaxcmr.helpers import make_subject_trial_masks
from jaxcmr.loss.transform_sequence_likelihood import (
    ExcludeTerminationLikelihoodLoss,
)
from jaxcmr.typing import FittingAlgorithm, FitResult


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


class DynamicQuadraticLoss:
    def __init__(self, model_factory_cls, dataset, features):
        self.target = jnp.asarray([0.25, 0.75])

    def __call__(self, trial_indices, base_params, free_param_names, x):
        target = self.target[: len(list(free_param_names))]
        return jnp.sum(jnp.square(x - target[:, None]), axis=0)


@jax.tree_util.register_pytree_node_class
class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = probability

    def tree_flatten(self):
        return (self.probability,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    def experience(self, item):
        return self

    def start_retrieving(self):
        return self

    def retrieve(self, item):
        return self

    def outcome_probability(self, item):
        return self.probability + 0.0 * item


class ConstantProbabilityFactory:
    def __init__(self, dataset, features):
        pass

    def create_trial_model(self, trial_index, parameters):
        return ConstantProbabilityModel(parameters["p"])


def make_minimal_dataset(subjects):
    return {"subject": np.asarray(subjects).reshape(-1, 1)}


def make_likelihood_dataset():
    return {
        "subject": np.asarray([1, 1]).reshape(-1, 1),
        "pres_itemnos": np.asarray([[1, 2], [1, 2]]),
        "recalls": np.asarray([[1, 0], [2, 0]]),
    }


def test_scipyde_constructor_parameter_names_match_fitting_protocol():
    """Behavior: ``ScipyDE`` exposes the constructor names required by the protocol."""
    protocol_params = inspect.signature(FittingAlgorithm.__init__).parameters
    scipy_params = inspect.signature(ScipyDE.__init__).parameters

    assert list(scipy_params)[:7] == list(protocol_params)[:7]
    assert "model_factory_cls" in scipy_params
    assert "loss_fn_cls" in scipy_params
    assert "model_factory" not in scipy_params
    assert "loss_fn" not in scipy_params
    assert "model_create_fn" not in scipy_params


def test_evosaxde_constructor_parameter_names_match_fitting_protocol():
    """Behavior: ``EvosaxDE`` exposes the constructor names required by the protocol."""
    protocol_params = inspect.signature(FittingAlgorithm.__init__).parameters
    evosax_params = inspect.signature(EvosaxDE.__init__).parameters

    assert list(evosax_params)[:7] == list(protocol_params)[:7]
    assert "model_factory_cls" in evosax_params
    assert "loss_fn_cls" in evosax_params
    assert "model_factory" not in evosax_params
    assert "loss_fn" not in evosax_params
    assert "model_create_fn" not in evosax_params


def test_scanevosaxde_constructor_parameter_names_match_fitting_protocol():
    """Behavior: ``ScanEvosaxDE`` exposes the constructor names required by the protocol."""
    protocol_params = inspect.signature(FittingAlgorithm.__init__).parameters
    batched_params = inspect.signature(ScanEvosaxDE.__init__).parameters

    assert list(batched_params)[:7] == list(protocol_params)[:7]
    assert "model_factory_cls" in batched_params
    assert "loss_fn_cls" in batched_params
    assert "model_factory" not in batched_params
    assert "loss_fn" not in batched_params
    assert "model_create_fn" not in batched_params


def test_mapevosaxde_constructor_parameter_names_match_fitting_protocol():
    """Behavior: ``MapEvosaxDE`` exposes the constructor names required by the protocol."""
    protocol_params = inspect.signature(FittingAlgorithm.__init__).parameters
    mapped_params = inspect.signature(MapEvosaxDE.__init__).parameters

    assert list(mapped_params)[:7] == list(protocol_params)[:7]
    assert "model_factory_cls" in mapped_params
    assert "loss_fn_cls" in mapped_params
    assert "model_factory" not in mapped_params
    assert "loss_fn" not in mapped_params
    assert "model_create_fn" not in mapped_params


def test_fitresult_requires_iteration_metadata():
    """Behavior: ``FitResult`` requires optimizer iteration metadata."""
    assert "nit" in FitResult.__required_keys__
    assert "converged" in FitResult.__required_keys__


def test_fitting_package_does_not_reexport_subject_mask_helper():
    """Behavior: subject-mask helpers are public from ``jaxcmr.helpers``."""
    assert "make_subject_trial_masks" not in fitting.__all__
    assert not hasattr(fitting, "make_subject_trial_masks")


def test_batchevosaxde_remains_scan_compatibility_alias():
    """Behavior: ``BatchEvosaxDE`` remains available for old artifacts."""
    assert BatchEvosaxDE is ScanEvosaxDE


def test_scipyde_passes_explicit_differential_evolution_controls(monkeypatch):
    """Behavior: ``ScipyDE`` forwards SciPy-parity control hyperparameters."""
    calls = []

    def fake_differential_evolution(func, bounds, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(fun=0.0, x=np.array([0.25]), nit=3, success=True)

    monkeypatch.setattr(
        scipy_fitting, "differential_evolution", fake_differential_evolution
    )
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

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

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


def test_scipyde_normalizes_scalar_diff_w(monkeypatch):
    """Behavior: ``ScipyDE`` normalizes scalar ``diff_w`` to a tuple."""
    calls = []

    def fake_differential_evolution(func, bounds, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(fun=0.0, x=np.array([0.25]), nit=3, success=True)

    monkeypatch.setattr(
        scipy_fitting, "differential_evolution", fake_differential_evolution
    )
    fitter = ScipyDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "diff_w": 0.85,
            "progress_bar": False,
        },
    )

    fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert fitter.all_hyperparams["diff_w"] == (0.85, 0.85)
    assert calls[0]["mutation"] == (0.85, 0.85)


def test_scipyde_uses_dithered_diff_w_by_default(monkeypatch):
    """Behavior: ``ScipyDE`` defaults to the current shared dither range."""
    calls = []

    def fake_differential_evolution(func, bounds, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(fun=0.0, x=np.array([0.25]), nit=3, success=True)

    monkeypatch.setattr(
        scipy_fitting, "differential_evolution", fake_differential_evolution
    )
    fitter = ScipyDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "progress_bar": False,
        },
    )

    fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert fitter.all_hyperparams["diff_w"] == (0.5, 1.0)
    assert calls[0]["mutation"] == (0.5, 1.0)


def test_scipyde_seed_is_deterministic():
    """Behavior: two ``ScipyDE`` instances with the same seed fit identically."""
    hyperparams = {
        "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "num_steps": 5,
        "pop_size": 4,
        "init": "random",
        "polish": False,
        "progress_bar": False,
        "seed": 123,
    }
    first = ScipyDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    second = ScipyDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )

    first_result = first.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]
    second_result = second.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert first_result["fitness"] == second_result["fitness"]
    assert first_result["nit"] == second_result["nit"]
    np.testing.assert_allclose(
        first_result["fits"]["x"], second_result["fits"]["x"]
    )
    np.testing.assert_allclose(
        first_result["fits"]["y"], second_result["fits"]["y"]
    )


def test_reference_loss_matches_scipy_adapter():
    """Behavior: canonical loss matches the callable SciPy adapter."""
    loss = ExcludeTerminationLikelihoodLoss(
        ConstantProbabilityFactory,
        make_likelihood_dataset(),
        None,
    )
    trial_indices = jnp.asarray([0, 1])
    base_params = {}
    free_param_names = {"p": [0.0, 1.0]}
    scipy_loss_fn = make_scipy_loss_fn(
        loss, trial_indices, base_params, free_param_names
    )
    samples = jnp.asarray([[0.25, 0.5, 0.75]])

    expected = np.asarray(scipy_loss_fn(samples))
    actual = np.asarray(loss(trial_indices, base_params, free_param_names, samples))
    single_expected = np.asarray(scipy_loss_fn(jnp.asarray([0.5])))
    single_actual = np.asarray(
        loss(trial_indices, base_params, free_param_names, jnp.asarray([[0.5]]))[0]
    )

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(single_actual, single_expected)


def test_evosaxde_fit_returns_finite_bounded_result():
    """Behavior: ``EvosaxDE.fit`` returns one finite bounded fit."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "num_steps": 25,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
        },
    )

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert np.isfinite(result["fitness"][0])
    assert result["fit_time"] >= 0.0
    assert len(result["nit"]) == 1
    assert len(result["converged"]) == 1
    assert result["fits"]["subject"] == [1]
    assert 0.0 <= result["fits"]["x"][0] <= 1.0
    assert 0.0 <= result["fits"]["y"][0] <= 1.0


def test_evosaxde_uses_canonical_loss_call():
    """Behavior: ``EvosaxDE`` uses canonical dynamic trial-index loss."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        DynamicQuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "num_steps": 10,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
        },
    )

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert np.isfinite(result["fitness"][0])
    assert result["fits"]["subject"] == [1]
    assert 0.0 <= result["fits"]["x"][0] <= 1.0
    assert 0.0 <= result["fits"]["y"][0] <= 1.0


def test_evosaxde_accepts_tuple_diff_w():
    """Behavior: tuple ``diff_w`` enables generation-level mutation dithering."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "num_steps": 5,
            "pop_size": 5,
            "diff_w": (0.5, 1.0),
            "progress_bar": False,
            "seed": 0,
        },
    )

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert fitter.all_hyperparams["diff_w"] == (0.5, 1.0)
    assert np.isfinite(result["fitness"][0])
    assert result["fit_time"] >= 0.0
    assert len(result["nit"]) == 1
    assert len(result["converged"]) == 1
    assert 0.0 <= result["fits"]["x"][0] <= 1.0
    assert 0.0 <= result["fits"]["y"][0] <= 1.0


def test_evosaxde_normalizes_scalar_diff_w():
    """Behavior: scalar ``diff_w`` is normalized to an inactive dither range."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 4,
            "diff_w": 0.85,
            "progress_bar": False,
            "seed": 0,
        },
    )

    assert fitter.all_hyperparams["diff_w"] == (0.85, 0.85)


def test_evosaxde_uses_dithered_diff_w_by_default():
    """Behavior: ``EvosaxDE`` defaults to the current shared dither range."""
    fitter = EvosaxDE(
        make_minimal_dataset([1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 1,
            "pop_size": 4,
            "progress_bar": False,
            "seed": 0,
        },
    )

    assert fitter.all_hyperparams["diff_w"] == (0.5, 1.0)


def test_evosaxde_uses_random_initialization_by_default():
    """Behavior: ``EvosaxDE`` samples the initial population uniformly by default."""
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

    key = jax.random.PRNGKey(0)
    population = np.asarray(fitter._sample_initial_population(key))
    expected = np.asarray(
        jax.random.uniform(
            key,
            (
                fitter.all_hyperparams["population_size"],
                fitter.num_parameters,
            ),
        )
    )

    np.testing.assert_allclose(population, expected)


def test_evosaxde_can_use_latin_hypercube_initialization():
    """Behavior: ``init='latinhypercube'`` samples one value per parameter stratum."""
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
            "init": "latinhypercube",
            "progress_bar": False,
            "seed": 0,
        },
    )

    population = np.asarray(fitter._sample_initial_population(jax.random.PRNGKey(0)))
    population_size = fitter.all_hyperparams["population_size"]
    lower_edges = np.arange(population_size) / population_size
    upper_edges = np.arange(1, population_size + 1) / population_size

    for column in range(population.shape[1]):
        sorted_values = np.sort(population[:, column])
        assert np.all(sorted_values >= lower_edges)
        assert np.all(sorted_values < upper_edges)


def test_evosaxde_resamples_out_of_bounds_candidates_by_default():
    """Behavior: ``EvosaxDE`` repairs out-of-bounds trial values like SciPy."""
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

    repaired = np.asarray(
        fitter._ensure_bounds(jax.random.PRNGKey(0), population)
    )

    assert np.all(repaired >= 0.0)
    assert np.all(repaired <= 1.0)
    assert not np.allclose(repaired[0], [0.0, 1.0])
    np.testing.assert_allclose(repaired[1], [0.25, 0.75])


def test_evosaxde_can_clip_out_of_bounds_candidates_for_compatibility():
    """Behavior: ``boundary_handling='clip'`` preserves the old repair behavior."""
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

    repaired = np.asarray(
        fitter._ensure_bounds(jax.random.PRNGKey(0), population)
    )

    np.testing.assert_allclose(repaired[0], [0.0, 1.0])
    np.testing.assert_allclose(repaired[1], [0.25, 0.75])


def test_evosaxde_fit_subjects_returns_one_result_per_subject():
    """Behavior: ``EvosaxDE.fit_subjects`` aggregates independent subject fits."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
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

    result = fitter.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert result["fits"]["subject"] == [1, 2]
    assert len(result["fitness"]) == 2
    assert len(result["nit"]) == 2
    assert len(result["converged"]) == 2
    assert len(result["fits"]["x"]) == 2
    assert all(np.isfinite(value) for value in result["fitness"])


def test_scanevosaxde_fit_subjects_returns_one_result_per_subject():
    """Behavior: ``ScanEvosaxDE.fit_subjects`` scans equal-length subject fits."""
    fitter = ScanEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
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

    result = fitter.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert result["fits"]["subject"] == [1, 2]
    assert len(result["fitness"]) == 2
    assert len(result["nit"]) == 2
    assert len(result["converged"]) == 2
    assert len(result["fits"]["x"]) == 2
    assert all(np.isfinite(value) for value in result["fitness"])


def test_scanevosaxde_seed_is_deterministic():
    """Behavior: two ``ScanEvosaxDE`` instances with the same seed fit identically."""
    hyperparams = {
        "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "num_steps": 10,
        "pop_size": 5,
        "progress_bar": False,
        "seed": 123,
    }
    first = ScanEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )
    second = ScanEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams=hyperparams,
    )

    first_result = first.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]
    second_result = second.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert first_result["fitness"] == second_result["fitness"]
    assert first_result["nit"] == second_result["nit"]
    assert first_result["converged"] == second_result["converged"]
    np.testing.assert_allclose(
        first_result["fits"]["x"], second_result["fits"]["x"]
    )
    np.testing.assert_allclose(
        first_result["fits"]["y"], second_result["fits"]["y"]
    )


def test_scanevosaxde_matches_sequential_evosaxde_on_equal_counts():
    """Behavior: scanned and sequential EvoSax agree on equal-length subjects."""
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

    sequential_result = sequential.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]
    batched_result = scanned.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert batched_result["fits"]["subject"] == sequential_result["fits"]["subject"]
    assert batched_result["nit"] == sequential_result["nit"]
    assert batched_result["converged"] == sequential_result["converged"]
    np.testing.assert_allclose(
        batched_result["fitness"], sequential_result["fitness"]
    )
    np.testing.assert_allclose(
        batched_result["fits"]["x"], sequential_result["fits"]["x"]
    )
    np.testing.assert_allclose(
        batched_result["fits"]["y"], sequential_result["fits"]["y"]
    )


def test_mapevosaxde_matches_sequential_evosaxde_on_equal_counts():
    """Behavior: mapped and sequential EvoSax agree on equal-length subjects."""
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

    sequential_result = sequential.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]
    mapped_result = mapped.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert mapped_result["fits"]["subject"] == sequential_result["fits"]["subject"]
    assert mapped_result["nit"] == sequential_result["nit"]
    assert mapped_result["converged"] == sequential_result["converged"]
    np.testing.assert_allclose(
        mapped_result["fitness"], sequential_result["fitness"]
    )
    np.testing.assert_allclose(
        mapped_result["fits"]["x"], sequential_result["fits"]["x"]
    )
    np.testing.assert_allclose(
        mapped_result["fits"]["y"], sequential_result["fits"]["y"]
    )
    assert mapped_result["hyperparameters"]["map_batch_size"] == 2


def test_mapevosaxde_accepts_zero_batch_size():
    """Behavior: ``map_batch_size=0`` requests the full-vmap lax.map path."""
    fitter = MapEvosaxDE(
        make_minimal_dataset([1, 1, 2, 2]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 2,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "map_batch_size": 0,
        },
    )

    result = fitter.fit_subjects(np.array([True, True, True, True]))  # type: ignore[arg-type]

    assert result["fits"]["subject"] == [1, 2]
    assert result["hyperparameters"]["map_batch_size"] == 0
    assert all(np.isfinite(value) for value in result["fitness"])


def test_mapevosaxde_rejects_negative_batch_size():
    """Behavior: ``map_batch_size`` must be omitted or non-negative."""
    with pytest.raises(ValueError, match="non-negative integer"):
        MapEvosaxDE(
            make_minimal_dataset([1, 1, 2, 2]),
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
                "map_batch_size": -1,
            },
        )


def test_scanevosaxde_requires_equal_trial_counts():
    """Behavior: ``ScanEvosaxDE`` raises clearly on unequal selected trial counts."""
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

    with pytest.raises(
        ValueError,
        match="same number of selected trials per subject",
    ):
        fitter.fit_subjects(np.array([True, True, True, True, True]))  # type: ignore[arg-type]


def test_evosaxde_replaces_nonfinite_population_fitness():
    """Behavior: non-finite candidate losses are converted to a finite penalty."""
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
        fitter._evaluate_trial_population(jnp.asarray([0]), population)
    )

    assert fitness[0] == 12345.0
    assert np.all(np.isfinite(fitness))


def test_evosaxde_convergence_matches_scipy_formula():
    """Behavior: convergence uses SciPy's population-energy tolerance rule."""
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


def test_evosaxde_absolute_tolerance_affects_convergence():
    """Behavior: absolute tolerance can converge when relative tolerance is strict."""
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
            "relative_tolerance": 0.0,
            "absolute_tolerance": 2.0,
        },
    )

    assert bool(fitter._has_converged(jnp.asarray([9.0, 10.0, 11.0])))


def test_evosaxde_penalized_population_does_not_converge():
    """Behavior: penalty replacement values do not satisfy convergence."""
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
    """Behavior: loose convergence tolerance skips remaining scan generations."""
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

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert result["converged"] == [True]
    assert result["nit"][0] < 10


def test_evosaxde_strict_tolerance_runs_all_steps():
    """Behavior: strict convergence tolerance runs until ``num_steps``."""
    fitter = EvosaxDE(
        make_minimal_dataset([1, 1, 1]),
        None,
        {},
        DummyModelFactory,
        QuadraticLoss,
        hyperparams={
            "bounds": {"x": [0.0, 1.0]},
            "num_steps": 5,
            "pop_size": 5,
            "progress_bar": False,
            "seed": 0,
            "relative_tolerance": 0.0,
            "absolute_tolerance": 0.0,
        },
    )

    result = fitter.fit(np.array([True, True, True]), subject_id=1)  # type: ignore[arg-type]

    assert result["converged"] == [False]
    assert result["nit"] == [5]


def test_single_subject_returns_one_mask():
    """Behavior: ``make_subject_trial_masks`` returns one mask for a single subject.

    Given:
      - A trial mask of all True and a subject vector with one unique subject.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Exactly one mask is returned, and it matches the input trial mask.
    Why this matters:
      - The single-subject case is the simplest partition; the mask
        should pass through unchanged.
    """
    # Arrange / Given
    trial_mask = np.array([True, True, True])
    subject_vector = np.array([1, 1, 1])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 1
    assert len(unique_subjects) == 1
    np.testing.assert_array_equal(masks[0], [True, True, True])


def test_two_subjects_partition_trials():
    """Behavior: ``make_subject_trial_masks`` partitions trials by subject.

    Given:
      - A trial mask of all True and a subject vector with two subjects.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Two masks are returned, each selecting only its subject's trials,
        and their union covers all trials.
    Why this matters:
      - Per-subject fitting depends on correct partitioning so each
        subject's data is isolated.
    """
    # Arrange / Given
    trial_mask = np.array([True, True, True, True])
    subject_vector = np.array([1, 1, 2, 2])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 2
    np.testing.assert_array_equal(unique_subjects, [1, 2])
    np.testing.assert_array_equal(masks[0], [True, True, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, True, True])


def test_false_trials_excluded_from_all_masks():
    """Behavior: ``make_subject_trial_masks`` respects False in the trial mask.

    Given:
      - A trial mask with some False entries and two subjects.
    When:
      - ``make_subject_trial_masks`` is called.
    Then:
      - Trials marked False are excluded from every subject mask.
    Why this matters:
      - Condition filtering via the trial mask must propagate into
        per-subject masks to avoid fitting on excluded trials.
    """
    # Arrange / Given
    trial_mask = np.array([True, False, True, False])
    subject_vector = np.array([1, 1, 2, 2])

    # Act / When
    masks, unique_subjects = make_subject_trial_masks(trial_mask, subject_vector)  # type: ignore[arg-type]

    # Assert / Then
    assert len(masks) == 2
    np.testing.assert_array_equal(masks[0], [True, False, False, False])
    np.testing.assert_array_equal(masks[1], [False, False, True, False])
