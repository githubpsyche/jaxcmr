"""Experimental CMR preparation and fitting retain their selected observations."""

from support import CMR_PARAMS, OPTIMIZED_CMR_PARAMS, recall_fixture
import logging
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.experimental.optimized_cmr_fitting import (
    EvosaxDE,
    ScipyTrustConstr,
    ScipyLBFGSB,
    ScipyDE,
)
from jaxcmr.loss.sequence_likelihood import ExcludeTerminationLikelihoodLoss
from jaxcmr.models.cmr import make_factory
from jaxcmr.experimental.optimized_cmr import (
    ModelFactory as OptimizedFactory,
    _sequence_likelihood,
)

FACTORY = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
)
PARAMS = dict(CMR_PARAMS)
OPTIMIZED_PARAMS = dict(OPTIMIZED_CMR_PARAMS)
NAMES = tuple((name for name, value in PARAMS.items() if not isinstance(value, bool)))


@pytest.fixture(autouse=True)
def precision():
    with jax.enable_x64(True):
        yield


def observations():
    return recall_fixture()


@pytest.mark.parametrize(
    "names", [NAMES, NAMES[::-1], ("choice_sensitivity", "primacy_decay")]
)
def test_planned_cmr_preserves_values_gradients_multiplicity_and_empty_data(names):
    data = observations()
    fast = OptimizedFactory(data, None)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    selections = [jnp.array([0]), jnp.array([1, 1]), jnp.array([2])]
    plans = fast.prepare_prediction_selections(
        selections, [data["recalls"][indices] != 0 for indices in selections]
    )

    def score(p, prepared):
        return _sequence_likelihood(prepared, OPTIMIZED_PARAMS, names, p[:, None])[0]

    evaluate = jax.jit(jax.value_and_grad(score))
    point = jnp.asarray([PARAMS[name] for name in names])
    for index in (0, 1, 2, 0):
        expected = jax.jit(
            jax.value_and_grad(
                lambda p, selected: original(selected, PARAMS, names, p[:, None])[0]
            )
        )(point, selections[index])
        actual = evaluate(point, plans[index])
        np.testing.assert_allclose(
            actual[0],
            expected[0],
            rtol=1e-11,
            atol=1e-10,
            err_msg="Prepared selections must preserve their observations",
        )
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-08, atol=1e-09)
    actual = evaluate(jnp.ones_like(point), plans[2])
    np.testing.assert_array_equal(actual[0], 0.0)
    np.testing.assert_array_equal(actual[1], jnp.zeros_like(point))


def test_mixed_geometries_empty_selections_and_source_mutation():
    data = {key: np.asarray(value).copy() for key, value in observations().items()}
    fast = OptimizedFactory(data, None)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    selections = [
        jnp.array([0]),
        jnp.array([3]),
        jnp.array([], dtype=jnp.int32),
        jnp.array([1, 0]),
    ]
    plans = fast.prepare_prediction_selections(
        selections, [data["recalls"][indices] != 0 for indices in selections]
    )
    point = jnp.array([[2.0]])
    expected = [
        np.asarray(original(indices, PARAMS, ("choice_sensitivity",), point))
        for indices in selections
    ]
    data["recalls"][:] = 0
    for plan, value in zip(plans, expected, strict=True):
        np.testing.assert_allclose(
            jax.jit(
                lambda x, bound: _sequence_likelihood(
                    bound, OPTIMIZED_PARAMS, ("choice_sensitivity",), x
                )
            )(point, plan),
            value,
            atol=1e-10,
        )
    assert fast.prepare_prediction_selections([], []) == ()


def planned_fitter(kind):
    data = observations()
    options = {
        "bounds": {"choice_sensitivity": [0.1, 5.0]},
        "num_steps": 2,
        "best_of": 1,
    }
    cls = {
        "gradient": ScipyTrustConstr,
        "lbfgsb": ScipyLBFGSB,
        "evosax": EvosaxDE,
        "scipy": ScipyDE,
    }[kind]
    if kind in ("gradient", "lbfgsb"):
        options.update(
            screening_size=4,
            screening_batch_size=4,
            initial_params={"choice_sensitivity": 2.0},
        )
    else:
        options.update(pop_size=4, progress_bar=False, polish=False)
    fitter = cls(data, OPTIMIZED_PARAMS, options)
    masks = [
        jnp.array([True, False, False, False]),
        jnp.array([False, True, True, False]),
    ]
    assert fitter.prepare_selections(masks) >= 0
    return fitter, masks, ExcludeTerminationLikelihoodLoss(FACTORY, data, None)


def fit_selection(fitter, kind, mask):
    if kind in ("gradient", "lbfgsb"):
        fitter.warmup(mask)
        return fitter.fit(mask)
    if kind == "evosax":
        return fitter.fit_once(mask, jax.random.PRNGKey(31))
    return fitter.fit(mask)


@pytest.mark.parametrize("kind", ["gradient", "lbfgsb", "evosax", "scipy"])
def test_fitters_preserve_observations_across_selection_changes_and_replanning(kind):
    fitter, masks, original = planned_fitter(kind)

    def check(mask):
        result = fit_selection(fitter, kind, mask)
        point = jnp.array([[result["fits"]["choice_sensitivity"][0]]])
        np.testing.assert_allclose(
            result["fitness"],
            original(jnp.flatnonzero(mask), PARAMS, ("choice_sensitivity",), point),
            atol=1e-10,
        )

    for mask in (masks[0], masks[1], masks[0]):
        check(mask)
    check(masks[0])
    fitter.prepare_selections([masks[1]])
    check(masks[1])
    fitter.prepare_selections([])
    check(jnp.array([False, False, False, True]))


@pytest.mark.audit
@pytest.mark.parametrize("kind", ["gradient", "evosax"])
def test_fitter_selection_switches_reuse_compilation(kind, caplog):
    fitter, masks, _ = planned_fitter(kind)
    fit_selection(fitter, kind, masks[0])
    caplog.clear()
    for mask in (masks[1], masks[0]):
        with caplog.at_level(logging.WARNING, logger="jax"), jax.log_compiles(True):
            fit_selection(fitter, kind, mask)
    names = (
        ("scalar_loss", "screen_values") if kind == "gradient" else ("_run_prepared",)
    )
    assert not [
        record.message
        for record in caplog.records
        if any(record.message.startswith(f"Compiling jit({name})") for name in names)
    ]
