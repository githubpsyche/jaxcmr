"""Numerical and fitter regression checks using one Lohnas-Kahana repetition trial."""

from support import CMR_PARAMS
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.fitting import ParameterSpace, ScipyDE, select_starts
from jaxcmr.fitting.gradient import ScipyLBFGSB
from jaxcmr.helpers import make_dataset
from jaxcmr.loss.transform_sequence_likelihood import ExcludeTerminationLikelihoodLoss
from jaxcmr.math import normalize_magnitude, relative_power_scale
from jaxcmr.models.cmr import make_factory

FACTORY = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
)
PARAMS = dict(CMR_PARAMS)


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def dataset():
    return make_dataset(
        jnp.array([[31, 29] + [0] * 38]),
        pres_itemnos=jnp.repeat(jnp.arange(1, 21), 2)[None, :],
        listLength=40,
        subject=2,
    )


def fitter(**options):
    return ScipyLBFGSB(
        dataset(),
        None,
        PARAMS,
        FACTORY,
        ExcludeTerminationLikelihoodLoss,
        {
            "bounds": {"choice_sensitivity": [0.1, 5.0]},
            "num_steps": 60,
            "best_of": 1,
            "screening_size": 0,
            "initial_params": {"choice_sensitivity": 2.0},
            **options,
        },
    )


@pytest.mark.parametrize("sensitivity", (1.0, 2.0, 99.0))
def test_relative_power_gradients_ignore_structural_zeros_and_remain_correct_at_one(sensitivity):
    values = jnp.array([1.0, 0.5, 0.0])
    fn = jax.jit(lambda sensitivity: relative_power_scale(values, sensitivity).sum())
    value, gradient = jax.value_and_grad(fn)(sensitivity)
    np.testing.assert_allclose(value, 1 + 0.5**sensitivity)
    np.testing.assert_allclose(gradient, 0.5**sensitivity * np.log(0.5), atol=1e-14)
    zeros = jnp.zeros(3)
    np.testing.assert_array_equal(relative_power_scale(zeros, 2.0), zeros)
    np.testing.assert_array_equal(jax.jacrev(relative_power_scale)(zeros, 2.0), np.zeros((3, 3)))


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_zero_norm_has_finite_explicit_zero_tangent_under_batched_reverse_mode(scale):
    values = scale * jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    result = jax.jit(jax.vmap(normalize_magnitude))(values)
    np.testing.assert_allclose(result, [[0, 0, 0], [0.6, 0.8, 0]])
    derivative = jax.jit(jax.jacrev(jax.vmap(normalize_magnitude)))(values)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(derivative[0], np.zeros((3, 2, 3)))


def test_normalization_does_not_hide_nonfinite_inputs():
    assert np.all(np.isnan(normalize_magnitude(jnp.array([np.nan, 1.0]))))


@pytest.mark.parametrize("sensitivity", [1.0, 2.0])
def test_cmr_gradients_match_finite_differences_on_actual_repeated_presentations(sensitivity):
    data = dataset()
    params = {**PARAMS, "choice_sensitivity": sensitivity}
    names = tuple((key for key, value in params.items() if not isinstance(value, bool)))
    point = jnp.array([params[name] for name in names])
    loss = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)

    @jax.jit
    def objective(x):
        return loss(jnp.array([0]), params, names, x[:, None])[0]

    value, derivative = jax.jit(jax.value_and_grad(objective))(point)
    offsets = jnp.eye(len(names)) * 1e-05
    numerical = jax.vmap(
        lambda delta: (objective(point + delta) - objective(point - delta)) / 2e-05
    )(offsets)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, numerical, rtol=1e-05, atol=1e-08)
    if sensitivity == 2.0:
        np.testing.assert_allclose(value, 5.9108604179346145, rtol=0, atol=1e-12)
    model = FACTORY(data, None).create_trial_model(jnp.array(0), params)
    for item in np.asarray(data["pres_itemnos"])[0]:
        model = model.experience(int(item))
    model = model.start_retrieving()
    manual = -jnp.log(model.outcome_probability(16))
    model = model.retrieve(16)
    manual -= jnp.log(model.outcome_probability(15))
    np.testing.assert_allclose(value, manual, atol=1e-12)


def test_one_parameter_fit_is_reproducible_bounded_and_improves_likelihood():
    mask = jnp.array([True])
    first = fitter(best_of=2, seed=17)
    first.warmup(mask)
    result = first.fit(mask, subject_id=2)
    again = fitter(best_of=2, seed=17).fit(mask, subject_id=2)
    assert result["fitness"][0] < 5.9108604179346145
    assert 0.1 <= result["fits"]["choice_sensitivity"][0] <= 5
    np.testing.assert_array_equal(result["fitness"], again["fitness"])
    assert result["fits"] == again["fits"]
    assert result["fits"]["subject"] == [2]
    assert "diagnostics" in result
    assert len(result["diagnostics"]["restarts"]) == 2
    assert result["diagnostics"]["nfev"] > 0
    predicted = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
        jnp.array([0]),
        PARAMS,
        ("choice_sensitivity",),
        jnp.array([[result["fits"]["choice_sensitivity"][0]]]),
    )[0]
    np.testing.assert_allclose(result["fitness"], [predicted], atol=1e-12)
    json.dumps(result)


def test_time_limit_retains_a_valid_evaluation_without_claiming_convergence():
    algorithm = fitter(time_limit_seconds=1e-09, best_of=5)
    result = algorithm.fit(jnp.array([True]))
    assert np.isfinite(result["fitness"][0])
    assert result["converged"] == [False]
    assert "diagnostics" in result
    assert result["diagnostics"]["nfev"] == 1
    assert "Time budget" in result["diagnostics"]["restarts"][0]["message"]


def test_invalid_gradients_fail_clearly_instead_of_reporting_a_fit():
    broken_params = {**PARAMS, "encoding_drift_rate": np.nan}
    algorithm = ScipyLBFGSB(
        dataset(),
        None,
        broken_params,
        FACTORY,
        ExcludeTerminationLikelihoodLoss,
        {"bounds": {"choice_sensitivity": [0.1, 5.0]}},
    )
    with pytest.raises(ValueError, match="No finite loss/gradient pair"):
        algorithm.fit(jnp.array([True]))


def test_single_precision_returns_the_actual_evaluated_parameters():
    with jax.enable_x64(False):
        algorithm = fitter(relative_tolerance=1e-06)
        result = algorithm.fit(jnp.array([True]))
        fitted = result["fits"]["choice_sensitivity"][0]
        assert 0.1 <= fitted <= 5.0
        assert fitted == float(np.float32(fitted))
        assert "diagnostics" in result
        assert result["diagnostics"]["jax_dtype"] == "float32"
        checked = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
            jnp.array([0]), PARAMS, ("choice_sensitivity",), jnp.array([[fitted]])
        )[0]
        np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-05)


def test_single_precision_drift_bounds_remain_inside_singular_endpoints():
    with jax.enable_x64(False):
        eps = float(np.finfo(float).eps)
        algorithm = fitter(
            bounds={"encoding_drift_rate": [eps, 1.0 - eps]},
            initial_params={"encoding_drift_rate": 0.5},
        )
        endpoints = jax.vmap(algorithm._physical_parameters)(jnp.array([[0.0], [1.0]]))
        assert eps <= float(endpoints[0, 0]) < float(endpoints[1, 0]) <= 1.0 - eps


def test_retention_does_not_silently_truncate_bounds_in_single_precision():
    with jax.enable_x64(False), pytest.raises(ValueError, match="representable"):
        ParameterSpace(
            {"primacy_decay": [np.finfo(float).eps, 100.0]}, {"primacy_decay": "exp_neg"}
        )


def test_retention_likelihood_and_gradient_match_physical_cmr():
    algorithm = fitter(
        bounds={"primacy_decay": [np.finfo(float).eps, 100.0]},
        transforms={"primacy_decay": "exp_neg"},
        initial_params={"primacy_decay": 0.3},
    )
    indices = jnp.array([0])
    prepared = algorithm._losses.select(indices)
    unit = algorithm.parameter_space.to_unit(jnp.array([0.3]))
    value, derivative = algorithm._value_and_grad(unit, prepared)
    loss = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)
    physical_value = loss(indices, PARAMS, ("primacy_decay",), jnp.array([[0.3]]))[0]
    np.testing.assert_allclose(value, physical_value, atol=1e-12)
    step = 1e-05
    numerical = (
        algorithm._value_and_grad(unit + step, prepared)[0]
        - algorithm._value_and_grad(unit - step, prepared)[0]
    ) / (2 * step)
    np.testing.assert_allclose(derivative[0], numerical, rtol=1e-05, atol=1e-08)
    result = algorithm.fit(jnp.array([True]))
    fitted = result["fits"]["primacy_decay"][0]
    checked = loss(indices, PARAMS, ("primacy_decay",), jnp.array([[fitted]]))[0]
    np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-12)
    assert result["free"]["primacy_decay"] == [np.finfo(float).eps, 100.0]
    json.dumps(result, allow_nan=False)


def test_screened_selection_rejects_invalid_scores_and_near_duplicate_starts():
    candidates = np.array([[0.1, 0.1], [0.11, 0.11], [0.9, 0.9], [0.3, 0.3], [0.7, 0.7]])
    scores = np.array([1.0, 2.0, 3.0, np.nan, np.inf])
    np.testing.assert_array_equal(select_starts(candidates, scores, 3, 0.1), [0, 2])
    np.testing.assert_array_equal(select_starts(candidates, scores, 3, 0.0), [0, 1, 2])


def test_screened_fitting_is_reproducible_and_keeps_an_explicit_first_start():
    config = {"screening_size": 8, "screening_batch_size": 4, "best_of": 3, "seed": 27}
    first = fitter(**config).fit(jnp.array([True]))
    second = fitter(**config).fit(jnp.array([True]))
    assert first["fits"] == second["fits"]
    assert "diagnostics" in first and "diagnostics" in second
    screen = first["diagnostics"]["screening"]
    assert screen["evaluated"] == 8
    assert screen["selected_indices"][0] == 0
    assert screen["selected_indices"] == second["diagnostics"]["screening"]["selected_indices"]
    np.testing.assert_allclose(
        first["diagnostics"]["restarts"][0]["initial_unit"], [(2 - 0.1) / 4.9]
    )
    assert len(first["diagnostics"]["restarts"]) <= 3
    assert first["diagnostics"]["history"][-1]["value"] == first["fitness"][0]


def test_scipyde_transform_keeps_physical_bounds_and_return_values():
    algorithm = ScipyDE(
        dataset(),
        None,
        PARAMS,
        FACTORY,
        ExcludeTerminationLikelihoodLoss,
        {
            "bounds": {"primacy_decay": [0.01, 10.0]},
            "transforms": {"primacy_decay": "exp_neg"},
            "num_steps": 2,
            "pop_size": 5,
            "seed": 9,
            "polish": True,
            "init": np.array([[0.1], [0.5], [1.0], [2.0], [5.0]]),
        },
    )
    result = algorithm.fit(jnp.array([True]))
    value = result["fits"]["primacy_decay"][0]
    assert 0.01 <= value <= 10.0
    checked = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
        jnp.array([0]), PARAMS, ("primacy_decay",), jnp.array([[value]])
    )[0]
    np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-12)
    assert result["free"]["primacy_decay"] == [0.01, 10.0]
