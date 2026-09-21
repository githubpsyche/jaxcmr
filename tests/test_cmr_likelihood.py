"""Scientific parity and selection safety for the optimized regular-CMR loss."""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.fitting import EvosaxDE, ScipyTrustConstr
from jaxcmr.fitting.scipy import make_scipy_loss_fn
from jaxcmr.helpers import make_dataset
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
from jaxcmr.loss.transform_sequence_likelihood import (
    ExcludeTerminationLikelihoodLoss,
    MemorySearchLikelihoodLoss,
    mask_trailing_terminations,
)
from jaxcmr.models.cmr import make_factory
from jaxcmr.models.optimized_cmr import ModelFactory as OptimizedFactory

FACTORY = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
)
PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "choice_sensitivity": 2.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}
NAMES = tuple(name for name, value in PARAMS.items() if not isinstance(value, bool))


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def repeated_trial():
    # Actual LK14 row 82, with both occurrences of all 20 items retained.
    return make_dataset(
        jnp.array([[31, 29] + [0] * 38]),
        pres_itemnos=jnp.repeat(jnp.arange(1, 21), 2)[None, :],
        listLength=40,
        subject=2,
    )


def four_lists():
    path = Path(__file__).resolve().parents[1] / "data/LohnasKahana2014.h5"
    if not path.exists():
        pytest.skip("Local LK14 dataset is required for the mixed-list regression.")
    with h5py.File(path) as file:
        return {
            name: jnp.asarray(
                file["data"][name][()].T[[78, 82, 80, 65]], dtype=jnp.int32
            )
            for name in ("subject", "listLength", "pres_itemnos", "recalls")
        }


@pytest.mark.parametrize("sensitivity", [1.0, 2.0])
def test_fast_repeated_trial_matches_all_event_values_and_gradients(sensitivity):
    data = repeated_trial()
    parameters = {**PARAMS, "choice_sensitivity": sensitivity}
    point = jnp.asarray([parameters[name] for name in NAMES])
    canonical = MemorySearchLikelihoodLoss(
        FACTORY, data, None, mask_trailing_terminations
    )
    expected = jax.jit(
        lambda p: canonical.predict_trial(
            jnp.array(0),
            dict(zip(NAMES, p))
            | {"learn_after_context_update": False, "allow_repeated_recalls": False},
        )
    )(point)
    prediction = OptimizedFactory(data, None).prepare_predictions(
        jnp.array([0]), data["recalls"] != 0
    )
    actual = jax.jit(
        lambda p: prediction.probabilities(parameters | dict(zip(NAMES, p)))
    )(point)[0][0]
    np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-12, rtol=1e-12)
    assert actual.shape == (2,)
    fast = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    prepared = fast.prepare_trials(jnp.array([0]))
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    evaluate = lambda fn: jax.jit(jax.value_and_grad(fn))(point)
    reference = evaluate(
        lambda p: original(jnp.array([0]), parameters, NAMES, p[:, None])[0]
    )
    for fn in (
        lambda p: fast(jnp.array([0]), parameters, NAMES, p[:, None])[0],
        lambda p: prepared(parameters, NAMES, p[:, None])[0],
    ):
        value, gradient = evaluate(fn)
        np.testing.assert_allclose(value, reference[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(gradient, reference[1], atol=1e-10, rtol=1e-9)
        assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize(
    "names", [NAMES, NAMES[::-1], ("choice_sensitivity", "primacy_decay")]
)
def test_partial_reordered_parameters_and_duplicate_trial_selection(names):
    data = four_lists()
    indices = jnp.array([3, 0, 3], dtype=jnp.int32)
    fast = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    prepared = fast.prepare_trials(indices)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    point = jnp.asarray([PARAMS[name] for name in names])
    expected = jax.jit(
        jax.value_and_grad(lambda p: original(indices, PARAMS, names, p[:, None])[0])
    )(point)
    actual = jax.jit(
        jax.value_and_grad(lambda p: prepared(PARAMS, names, p[:, None])[0])
    )(point)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("width", [0, 40])
def test_no_recalls_and_empty_selection_have_zero_loss_and_gradient(width):
    data = {**repeated_trial(), "recalls": jnp.zeros((1, width), dtype=jnp.int32)}
    loss = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    point = jnp.asarray([PARAMS[name] for name in NAMES])
    for indices in (jnp.array([0]), jnp.array([], dtype=jnp.int32)):
        prepared = loss.prepare_trials(indices)
        value, gradient = jax.jit(
            jax.value_and_grad(lambda p, bound: bound(PARAMS, NAMES, p[:, None])[0])
        )(point, prepared)
        assert value == 0
        np.testing.assert_array_equal(gradient, jnp.zeros(len(NAMES)))


def test_preparation_snapshots_and_does_not_mutate_previous_selections():
    data = {name: np.asarray(value).copy() for name, value in four_lists().items()}
    loss = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    point = jnp.array([[2.0]])
    first = loss.prepare_trials(jnp.array([0]))
    second = loss.prepare_trials(jnp.array([1]))
    run_first = jax.jit(lambda p: first(PARAMS, ("choice_sensitivity",), p))
    before = run_first(point)
    data["recalls"][:] = 0
    run_second = jax.jit(lambda p: second(PARAMS, ("choice_sensitivity",), p))
    for fn, index in ((run_first, 0), (run_second, 1), (run_first, 0)):
        np.testing.assert_allclose(
            fn(point),
            original(jnp.array([index]), PARAMS, ("choice_sensitivity",), point),
            atol=1e-11,
        )
    np.testing.assert_array_equal(run_first(point), before)


def test_gradient_fitter_changes_same_size_trial_selections_safely():
    data = four_lists()
    algorithm = ScipyTrustConstr(
        data,
        None,
        PARAMS,
        OptimizedFactory,
        FastCMRLikelihoodLoss,
        {
            "bounds": {"choice_sensitivity": [0.1, 5.0]},
            "num_steps": 2,
            "best_of": 1,
            "screening_size": 0,
            "initial_params": {"choice_sensitivity": 2.0},
        },
    )
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    for index in (0, 1, 0):
        mask = jnp.arange(4) == index
        algorithm.warmup(mask)
        fitted = algorithm.fit(mask)
        point = jnp.array([[fitted["fits"]["choice_sensitivity"][0]]])
        np.testing.assert_allclose(
            fitted["fitness"],
            original(jnp.array([index]), PARAMS, ("choice_sensitivity",), point),
            atol=1e-11,
        )
        assert fitted["diagnostics"]["prepared_trial_count"] == 1


def test_serial_evosax_uses_prepared_selection_without_stale_jit_state():
    data = four_lists()
    algorithm = EvosaxDE(
        data,
        None,
        PARAMS,
        OptimizedFactory,
        FastCMRLikelihoodLoss,
        {
            "bounds": {"choice_sensitivity": [0.1, 5.0]},
            "num_steps": 2,
            "pop_size": 4,
            "best_of": 1,
            "progress_bar": False,
        },
    )
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    for index in (0, 1, 0):
        result = algorithm.fit_once(jnp.arange(4) == index, jax.random.PRNGKey(31))
        point = jnp.array([[result["fits"]["choice_sensitivity"][0]]])
        np.testing.assert_allclose(
            result["fitness"],
            original(jnp.array([index]), PARAMS, ("choice_sensitivity",), point),
            atol=1e-11,
        )
        assert algorithm._losses.prepared_trial_count == 1


def test_scipy_de_loss_preparation_preserves_scalar_and_population_interfaces():
    data = four_lists()
    indices = jnp.array([1, 3])
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    fast = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    function = make_scipy_loss_fn(fast, indices, PARAMS, ("choice_sensitivity",))
    values = np.array([[0.5, 1.0, 2.0]])
    np.testing.assert_allclose(
        function(values),
        original(indices, PARAMS, ("choice_sensitivity",), values),
        atol=1e-11,
    )
    np.testing.assert_allclose(function(values[:, 1]), function(values)[1], atol=1e-11)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pres_itemnos", jnp.zeros((1, 40), dtype=int)),
        ("listLength", jnp.array([[39]])),
        ("recalls", jnp.array([[31, 0, 29] + [0] * 37])),
        ("recalls", jnp.array([[31, 32] + [0] * 38])),
        ("recalls", jnp.array([[-1] + [0] * 39])),
        ("recalls", jnp.zeros((1, 40), dtype=float)),
    ],
)
def test_unsupported_observations_fail_explicitly(field, value):
    with pytest.raises(ValueError):
        FastCMRLikelihoodLoss(
            OptimizedFactory, {**repeated_trial(), field: value}, None
        )


def test_optimized_model_rejects_unsupported_features_and_parameters():
    data = repeated_trial()
    with pytest.raises(ValueError, match="features must be None"):
        FastCMRLikelihoodLoss(OptimizedFactory, data, jnp.eye(40))
    prepared = FastCMRLikelihoodLoss(OptimizedFactory, data, None).prepare_trials(
        jnp.array([0])
    )
    point = jnp.array([[2.0]])
    for flag in ("learn_after_context_update", "allow_repeated_recalls"):
        with pytest.raises(ValueError, match=flag):
            jax.jit(
                lambda p, flag=flag: prepared(
                    {**PARAMS, flag: True}, ("choice_sensitivity",), p
                )
            )(point)
    with pytest.raises(ValueError, match="parameter mismatch"):
        prepared(PARAMS, ("stop_probability_scale",), point)
