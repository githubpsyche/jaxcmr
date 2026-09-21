"""Numerical and fitter regression checks using one Lohnas-Kahana repetition trial."""

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
from jaxcmr.typing import FittingAlgorithm

FACTORY = make_factory(linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination)
PARAMS = {
    "encoding_drift_rate": .5, "start_drift_rate": .4, "recall_drift_rate": .6,
    "shared_support": 1., "item_support": 1.5, "learning_rate": .3,
    "primacy_scale": 1., "primacy_decay": .1, "choice_sensitivity": 2.,
    "learn_after_context_update": False, "allow_repeated_recalls": False,
}


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def dataset():
    # Raw LohnasKahana2014 row 82: 20 item labels, each presented twice;
    # observed position codes 31 and 29 map to item labels 16 and 15.
    # Source checksum and unmodified arrays are in the research execution record.
    return make_dataset(
        jnp.array([[31, 29] + [0] * 38]),
        pres_itemnos=jnp.repeat(jnp.arange(1, 21), 2)[None, :],
        listLength=40, subject=2,
    )


def fitter(**options):
    return ScipyLBFGSB(dataset(), None, PARAMS, FACTORY, ExcludeTerminationLikelihoodLoss, {
        "bounds": {"choice_sensitivity": [.1, 5.]}, "num_steps": 60,
        "best_of": 1, "screening_size": 0,
        "initial_params": {"choice_sensitivity": 2.}, **options,
    })


def test_relative_power_gradients_ignore_structural_zeros_and_remain_correct_at_one():
    values = jnp.array([1., .5, 0.])
    fn = jax.jit(lambda sensitivity: relative_power_scale(values, sensitivity).sum())
    for sensitivity in (1., 2., 99.):
        value, gradient = jax.value_and_grad(fn)(sensitivity)
        np.testing.assert_allclose(value, 1 + .5**sensitivity)
        np.testing.assert_allclose(gradient, .5**sensitivity * np.log(.5), atol=1e-14)
    zeros = jnp.zeros(3)
    np.testing.assert_array_equal(relative_power_scale(zeros, 2.), zeros)
    np.testing.assert_array_equal(jax.jacrev(relative_power_scale)(zeros, 2.), np.zeros((3, 3)))


def test_zero_norm_has_finite_explicit_zero_tangent_under_batched_reverse_mode():
    values = jnp.array([[0., 0., 0.], [3., 4., 0.]])
    result = jax.jit(jax.vmap(normalize_magnitude))(values)
    np.testing.assert_allclose(result, [[0, 0, 0], [.6, .8, 0]])
    derivative = jax.jit(jax.jacrev(jax.vmap(normalize_magnitude)))(values)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(derivative[0], np.zeros((3, 2, 3)))


def test_normalization_does_not_hide_nonfinite_inputs():
    assert np.all(np.isnan(normalize_magnitude(jnp.array([np.nan, 1.]))))


@pytest.mark.parametrize("sensitivity", [1., 2.])
def test_cmr_gradients_match_finite_differences_on_actual_repeated_presentations(sensitivity):
    data = dataset()
    params = {**PARAMS, "choice_sensitivity": sensitivity}
    names = tuple(key for key, value in params.items() if not isinstance(value, bool))
    point = jnp.array([params[name] for name in names])
    loss = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)

    @jax.jit
    def objective(x):
        return loss(jnp.array([0]), params, names, x[:, None])[0]

    value, derivative = jax.jit(jax.value_and_grad(objective))(point)
    offsets = jnp.eye(len(names)) * 1e-5
    numerical = jax.vmap(lambda delta: (objective(point + delta) - objective(point - delta)) / 2e-5)(offsets)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, numerical, rtol=1e-5, atol=1e-8)
    if sensitivity == 2.:
        np.testing.assert_allclose(value, 5.9108604179346145, rtol=0, atol=1e-12)
    # Independent sequence evaluation verifies position->item recoding and that
    # every repeated presentation still updates the model.
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
    assert .1 <= result["fits"]["choice_sensitivity"][0] <= 5
    np.testing.assert_array_equal(result["fitness"], again["fitness"])
    assert result["fits"] == again["fits"]
    assert result["fits"]["subject"] == [2]
    assert "diagnostics" in result
    assert len(result["diagnostics"]["restarts"]) == 2
    assert result["diagnostics"]["nfev"] > 0
    predicted = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
        jnp.array([0]), PARAMS, ("choice_sensitivity",),
        jnp.array([[result["fits"]["choice_sensitivity"][0]]]),
    )[0]
    np.testing.assert_allclose(result["fitness"], [predicted], atol=1e-12)
    json.dumps(result)


def test_time_limit_retains_a_valid_evaluation_without_claiming_convergence():
    algorithm = fitter(time_limit_seconds=1e-9, best_of=5)
    result = algorithm.fit(jnp.array([True]))
    assert np.isfinite(result["fitness"][0])
    assert result["converged"] == [False]
    assert "diagnostics" in result
    assert result["diagnostics"]["nfev"] == 1
    assert "Time budget" in result["diagnostics"]["restarts"][0]["message"]


def test_invalid_gradients_fail_clearly_instead_of_reporting_a_fit():
    broken_params = {**PARAMS, "encoding_drift_rate": np.nan}
    algorithm = ScipyLBFGSB(dataset(), None, broken_params, FACTORY, ExcludeTerminationLikelihoodLoss, {
        "bounds": {"choice_sensitivity": [.1, 5.]},
    })
    with pytest.raises(ValueError, match="No finite loss/gradient pair"):
        algorithm.fit(jnp.array([True]))


@pytest.mark.parametrize("options", [
    {"bounds": {}}, {"bounds": {"choice_sensitivity": [1, 1]}},
    {"best_of": 0}, {"initial_params": {"choice_sensitivity": 9}},
    {"initial_params": {}}, {"time_limit_seconds": -1},
])
def test_invalid_optimizer_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        fitter(**options)


def test_empty_or_nonboolean_trial_selection_is_rejected():
    algorithm = fitter()
    for mask in (jnp.array([False]), jnp.array([1]), jnp.array([True, False])):
        with pytest.raises(ValueError, match="trial_mask"):
            algorithm.fit(mask)


def test_fit_subjects_uses_the_common_fitter_contract():
    algorithm: FittingAlgorithm = fitter()
    result = algorithm.fit_subjects(jnp.array([True]))
    assert result["fits"]["subject"] == [2]
    assert len(result["fitness"]) == len(result["nit"]) == len(result["converged"]) == 1
    assert "diagnostics" in result
    assert len(result["diagnostics"]["subjects"]) == 1
    assert result["fitness"][0] < 5.9108604179346145


def test_single_precision_returns_the_actual_evaluated_parameters():
    with jax.enable_x64(False):
        algorithm = fitter(relative_tolerance=1e-6)
        result = algorithm.fit(jnp.array([True]))
        fitted = result["fits"]["choice_sensitivity"][0]
        assert .1 <= fitted <= 5.
        assert fitted == float(np.float32(fitted))
        assert "diagnostics" in result
        assert result["diagnostics"]["jax_dtype"] == "float32"
        checked = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
            jnp.array([0]), PARAMS, ("choice_sensitivity",), jnp.array([[fitted]]),
        )[0]
        np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-5)


def test_single_precision_drift_bounds_remain_inside_singular_endpoints():
    with jax.enable_x64(False):
        eps = float(np.finfo(float).eps)
        algorithm = fitter(bounds={"encoding_drift_rate": [eps, 1. - eps]}, initial_params={"encoding_drift_rate": .5})
        endpoints = jax.vmap(algorithm._physical_parameters)(jnp.array([[0.], [1.]]))
        assert eps <= float(endpoints[0, 0]) < float(endpoints[1, 0]) <= 1. - eps


def test_parameter_space_preserves_physical_values_under_jit_and_vmap():
    space = ParameterSpace({"d": [np.finfo(float).eps, 100.], "p": [.01, 10.], "b": [.1, .9]},
                           {"d": "exp_neg", "p": "log"})
    physical = jnp.array([[.3, 1., .4], [99., .02, .89], [np.finfo(float).eps, 10., .1]])
    unit = jax.jit(space.to_unit)(physical)
    np.testing.assert_allclose(jax.jit(jax.vmap(space.to_physical))(unit), physical, rtol=1e-13, atol=1e-14)
    endpoints = jax.jit(space.to_physical)(jnp.array([[0., 0., 0.], [1., 1., 1.]]))
    np.testing.assert_allclose(endpoints, [[100., .01, .1], [np.finfo(float).eps, 10., .9]], rtol=1e-13)
    assert np.all(np.asarray(endpoints) >= space.bounds[:, 0])
    assert np.all(np.asarray(endpoints) <= space.bounds[:, 1])


@pytest.mark.parametrize("bounds, transforms", [
    ({"d": [0., 1.]}, {"typo": "log"}),
    ({"d": [0., 1.]}, {"d": "log"}),
    ({"d": [0., 1.]}, {"d": "unknown"}),
    ({"d": [0., 1000.]}, {"d": "exp_neg"}),
])
def test_invalid_or_unrepresentable_transforms_fail_clearly(bounds, transforms):
    with pytest.raises(ValueError):
        ParameterSpace(bounds, transforms)


def test_retention_does_not_silently_truncate_bounds_in_single_precision():
    with jax.enable_x64(False), pytest.raises(ValueError, match="representable"):
        ParameterSpace({"primacy_decay": [np.finfo(float).eps, 100.]}, {"primacy_decay": "exp_neg"})


def test_retention_likelihood_and_gradient_match_physical_cmr():
    algorithm = fitter(bounds={"primacy_decay": [np.finfo(float).eps, 100.]},
                       transforms={"primacy_decay": "exp_neg"}, initial_params={"primacy_decay": .3})
    indices = jnp.array([0])
    prepared = algorithm._losses.select(indices)
    unit = algorithm.parameter_space.to_unit(jnp.array([.3]))
    value, derivative = algorithm._value_and_grad(unit, prepared)
    loss = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)
    physical_value = loss(indices, PARAMS, ("primacy_decay",), jnp.array([[.3]]))[0]
    np.testing.assert_allclose(value, physical_value, atol=1e-12)
    step = 1e-5
    numerical = (algorithm._value_and_grad(unit + step, prepared)[0]
                 - algorithm._value_and_grad(unit - step, prepared)[0]) / (2 * step)
    np.testing.assert_allclose(derivative[0], numerical, rtol=1e-5, atol=1e-8)
    result = algorithm.fit(jnp.array([True]))
    fitted = result["fits"]["primacy_decay"][0]
    checked = loss(indices, PARAMS, ("primacy_decay",), jnp.array([[fitted]]))[0]
    np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-12)
    assert result["free"]["primacy_decay"] == [np.finfo(float).eps, 100.]
    json.dumps(result, allow_nan=False)


def test_screened_selection_rejects_invalid_scores_and_near_duplicate_starts():
    candidates = np.array([[.1, .1], [.11, .11], [.9, .9], [.3, .3], [.7, .7]])
    scores = np.array([1., 2., 3., np.nan, np.inf])
    np.testing.assert_array_equal(select_starts(candidates, scores, 3, .1), [0, 2])
    np.testing.assert_array_equal(select_starts(candidates, scores, 3, 0.), [0, 1, 2])


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
    np.testing.assert_allclose(first["diagnostics"]["restarts"][0]["initial_unit"], [(2 - .1) / 4.9])
    assert len(first["diagnostics"]["restarts"]) <= 3
    assert first["diagnostics"]["history"][-1]["value"] == first["fitness"][0]


def test_screening_deadline_is_shared_with_local_solves():
    result = fitter(screening_size=8, screening_batch_size=4, best_of=3,
                    time_limit_seconds=1e-9).fit(jnp.array([True]))
    assert "diagnostics" in result
    assert result["diagnostics"]["screening"]["evaluated"] == 4
    assert result["diagnostics"]["nfev"] == 1
    assert result["converged"] == [False]


def test_scipyde_transform_keeps_physical_bounds_and_return_values():
    algorithm = ScipyDE(dataset(), None, PARAMS, FACTORY, ExcludeTerminationLikelihoodLoss, {
        "bounds": {"primacy_decay": [.01, 10.]}, "transforms": {"primacy_decay": "exp_neg"},
        "num_steps": 2, "pop_size": 5, "seed": 9, "polish": True,
        "init": np.array([[.1], [.5], [1.], [2.], [5.]]),
    })
    result = algorithm.fit(jnp.array([True]))
    value = result["fits"]["primacy_decay"][0]
    assert .01 <= value <= 10.
    checked = ExcludeTerminationLikelihoodLoss(FACTORY, dataset(), None)(
        jnp.array([0]), PARAMS, ("primacy_decay",), jnp.array([[value]]),
    )[0]
    np.testing.assert_allclose(result["fitness"][0], checked, atol=1e-12)
    assert result["free"]["primacy_decay"] == [.01, 10.]
