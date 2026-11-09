from support import CMR_PARAMS
from support import recall_fixture

"Data-switching, compilation reuse and loss compatibility for explicit plans."
import logging
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.fitting import EvosaxDE, ScipyTrustConstr
from jaxcmr.fitting._preparation import LossPreparation
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
from jaxcmr.loss.prepared import PreparedLoss
from jaxcmr.loss.transform_sequence_likelihood import ExcludeTerminationLikelihoodLoss
from jaxcmr.models.cmr import make_factory
from jaxcmr.models.optimized_cmr import ModelFactory as OptimizedFactory

FACTORY = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
)
PARAMS = dict(CMR_PARAMS)
NAMES = tuple((name for name, value in PARAMS.items() if not isinstance(value, bool)))


@pytest.fixture(autouse=True)
def precision():
    with jax.enable_x64(True):
        yield


def observations():
    return recall_fixture()


def quadratic(data, base, names, parameters):
    return jnp.sum((parameters - data[:, None]) ** 2, axis=0)


def test_prepared_container_shares_a_trace_without_cmr():
    traces = []

    def score(x, prepared):
        traces.append(1)
        return prepared({}, ("x",), x[:, None])[0]

    joint = jax.jit(jax.value_and_grad(score))
    first = PreparedLoss(jnp.array([1.0]), quadratic)
    second = PreparedLoss(jnp.array([3.0]), quadratic)
    for prepared, value, derivative in (
        (first, 1.0, -2.0),
        (second, 9.0, -6.0),
        (first, 1.0, -2.0),
    ):
        actual, gradient = joint(jnp.array([0.0]), prepared)
        np.testing.assert_allclose(actual, value)
        np.testing.assert_allclose(gradient, [derivative])
    assert len(traces) == 1


@pytest.mark.parametrize("batch_preparation", [False, True])
def test_ordinary_cmr_selections_share_compilation_and_preserve_gradients(batch_preparation):
    """Fallback A/B/A selections reuse code without specializing away observations."""
    loss = ExcludeTerminationLikelihoodLoss(FACTORY, observations(), None)
    selections = [jnp.array([0]), jnp.array([1]), jnp.array([0])]
    plans = (
        loss.prepare_selections(selections)
        if batch_preparation
        else tuple((loss.prepare_trials(indices) for indices in selections))
    )
    traces = []
    names = ("choice_sensitivity", "encoding_drift_rate")
    point = jnp.asarray([PARAMS[name] for name in names])

    def prepared_score(parameters, prepared):
        traces.append(1)
        return prepared(PARAMS, names, parameters[:, None])[0]

    evaluate = jax.jit(jax.value_and_grad(prepared_score))
    reference = jax.jit(
        jax.value_and_grad(
            lambda parameters, indices: loss(indices, PARAMS, names, parameters[:, None])[0]
        )
    )
    values = []
    for indices, plan in zip(selections, plans, strict=True):
        actual = evaluate(point, plan)
        expected = reference(point, indices)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-08, atol=1e-09)
        values.append(float(actual[0]))
    assert values[0] != values[1]
    assert values[0] == values[2]
    assert len(traces) == 1


@pytest.mark.parametrize("names", [NAMES, NAMES[::-1], ("choice_sensitivity", "primacy_decay")])
def test_planned_cmr_preserves_values_gradients_multiplicity_and_empty_data(names):
    data = observations()
    fast = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    selections = [jnp.array([0]), jnp.array([1, 1]), jnp.array([2])]
    plans = fast.prepare_selections(selections)
    traces = []

    def score(p, prepared):
        traces.append(1)
        return prepared(PARAMS, names, p[:, None])[0]

    evaluate = jax.jit(jax.value_and_grad(score))
    point = jnp.asarray([PARAMS[name] for name in names])
    for index in (0, 1, 2, 0):
        expected = jax.jit(
            jax.value_and_grad(lambda p, selected: original(selected, PARAMS, names, p[:, None])[0])
        )(point, selections[index])
        actual = evaluate(point, plans[index])
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-08, atol=1e-09)
    actual = evaluate(jnp.ones_like(point), plans[2])
    np.testing.assert_array_equal(actual[0], 0.0)
    np.testing.assert_array_equal(actual[1], jnp.zeros_like(point))
    assert len(traces) == 2


def test_mixed_geometries_empty_selections_and_source_mutation():
    data = {key: np.asarray(value).copy() for key, value in observations().items()}
    fast = FastCMRLikelihoodLoss(OptimizedFactory, data, None)
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)
    selections = [jnp.array([0]), jnp.array([3]), jnp.array([], dtype=jnp.int32), jnp.array([1, 0])]
    plans = fast.prepare_selections(selections)
    point = jnp.array([[2.0]])
    expected = [
        np.asarray(original(indices, PARAMS, ("choice_sensitivity",), point))
        for indices in selections
    ]
    data["recalls"][:] = 0
    for plan, value in zip(plans, expected, strict=True):
        np.testing.assert_allclose(
            jax.jit(lambda x, bound: bound(PARAMS, ("choice_sensitivity",), x))(point, plan),
            value,
            atol=1e-10,
        )
    assert fast.prepare_selections([]) == ()


@pytest.mark.parametrize("kind", ["gradient", "evosax"])
def test_fitters_reuse_planned_data_across_different_selection_sizes(kind, caplog):
    data = observations()
    options = {"bounds": {"choice_sensitivity": [0.1, 5.0]}, "num_steps": 2, "best_of": 1}
    cls = ScipyTrustConstr if kind == "gradient" else EvosaxDE
    if kind == "gradient":
        options.update(
            screening_size=4, screening_batch_size=4, initial_params={"choice_sensitivity": 2.0}
        )
    else:
        options.update(pop_size=4, progress_bar=False)
    fitter = cls(data, None, PARAMS, OptimizedFactory, FastCMRLikelihoodLoss, options)
    masks = [jnp.array([True, False, False, False]), jnp.array([False, True, True, False])]
    assert fitter.prepare_selections(masks) >= 0
    original = ExcludeTerminationLikelihoodLoss(FACTORY, data, None)

    def fit(mask):
        if kind == "gradient":
            fitter.warmup(mask)
            return fitter.fit(mask)
        return fitter.fit_once(mask, jax.random.PRNGKey(31))

    first = fit(masks[0])
    prepared = fitter._losses.select(jnp.flatnonzero(masks[0]))
    evaluate = fitter._value_and_grad if kind == "gradient" else fitter._compiled_run
    old = partial(evaluate, prepared=prepared)
    for mask in (masks[1], masks[0]):
        with caplog.at_level(logging.WARNING, logger="jax"), jax.log_compiles(True):
            result = fit(mask)
        point = jnp.array([[result["fits"]["choice_sensitivity"][0]]])
        np.testing.assert_allclose(
            result["fitness"],
            original(jnp.flatnonzero(mask), PARAMS, ("choice_sensitivity",), point),
            atol=1e-10,
        )
    names = ("scalar_loss", "screen_values") if kind == "gradient" else ("_run_generations",)
    assert not [
        r.message
        for r in caplog.records
        if any((r.message.startswith(f"Compiling jit({name})") for name in names))
    ]
    if kind == "gradient":
        unit = fitter.parameter_space.to_unit(jnp.array([2.0]))
        before = old(unit)
        fitter.prepare_selections([masks[1]])
        fit(masks[1])
        after = old(unit)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
    else:
        retained = old(jax.random.PRNGKey(31))
        np.testing.assert_allclose(retained[0], first["fitness"][0], atol=1e-10)
    fitter.prepare_selections([])
    mask = jnp.array([False, False, False, True])
    result = fit(mask)
    point = jnp.array([[result["fits"]["choice_sensitivity"][0]]])
    np.testing.assert_allclose(
        result["fitness"],
        original(jnp.array([3]), PARAMS, ("choice_sensitivity",), point),
        atol=1e-10,
    )


class OrdinaryLoss:
    def __init__(self, factory, dataset, features):
        pass

    def __call__(self, indices, base, names, parameters):
        return jnp.sum(parameters**2, axis=0)


class CallablePlanLoss(OrdinaryLoss):
    def prepare_trials(self, indices):
        target = float(np.asarray(indices).mean())
        return lambda base, names, parameters: jnp.sum((parameters - target) ** 2, axis=0)

    def prepare_selections(self, indices):
        return tuple((self.prepare_trials(selected) for selected in indices))


@pytest.mark.parametrize("fitter_cls", [ScipyTrustConstr, EvosaxDE])
def test_ordinary_prepared_callables_and_unsupported_planning_are_explicit(fitter_cls):
    data = observations()
    options = {
        "bounds": {"x": [-2.0, 4.0]},
        "screening_size": 0,
        "best_of": 1,
        "num_steps": 2,
        "pop_size": 4,
        "progress_bar": False,
    }
    generic = fitter_cls(data, None, {}, object, OrdinaryLoss, options)
    with pytest.raises(TypeError, match="does not support"):
        generic.prepare_selections([jnp.array([True, False, False, False])])
    assert np.isfinite(generic.fit(jnp.array([True, False, False, False]))["fitness"][0])
    planned = fitter_cls(data, None, {}, object, CallablePlanLoss, options)
    masks = [jnp.arange(4) == index for index in (0, 1)]
    planned.prepare_selections(masks)
    for index, mask in enumerate(masks):
        result = planned.fit(mask)
        np.testing.assert_allclose(result["fitness"][0], (result["fits"]["x"][0] - index) ** 2)
    old_loss = planned._losses.select(jnp.array([0]))
    with pytest.raises(ValueError, match="Boolean"):
        planned.prepare_selections([jnp.array([1, 0, 0, 0])])
    assert planned._losses.select(jnp.array([0])) is old_loss
    planned._losses.select(jnp.array([1]))
    assert planned._losses.select(jnp.array([0])) is old_loss


class IndexLoss(OrdinaryLoss):
    """An unhashable ordinary loss whose target depends on order and multiplicity."""

    __hash__ = None

    def __call__(self, indices, base, names, parameters):
        target = jnp.sum(indices * jnp.arange(1, len(indices) + 1))
        return jnp.sum((parameters - target) ** 2, axis=0)


class CallableIndexLoss(IndexLoss):
    def prepare_trials(self, indices):
        return lambda base, names, parameters: self(indices, base, names, parameters)


class ExplicitIndexLoss(IndexLoss):
    def prepare_trials(self, indices):
        return PreparedLoss(indices, self.__call__)


@pytest.mark.parametrize("loss_cls", [IndexLoss, CallableIndexLoss, ExplicitIndexLoss])
def test_unified_loss_preserves_dynamic_selection_and_retained_observations(loss_cls):
    losses = LossPreparation(loss_cls(None, None, None))
    traces = []

    @jax.jit
    def value_and_gradient(point, prepared):
        traces.append(1)
        return jax.value_and_grad(lambda x: prepared({}, ("x",), x[:, None])[0])(point)

    first = losses.select(jnp.array([2, 0, 2]))
    if loss_cls is not IndexLoss:
        assert losses.select(jnp.array([2, 0, 2])) is first
    second = losses.select(jnp.array([0, 2, 2]))
    for prepared, target in ((first, 8.0), (second, 10.0), (first, 8.0)):
        value, gradient = value_and_gradient(jnp.array([1.0]), prepared)
        np.testing.assert_allclose(value, (1.0 - target) ** 2)
        np.testing.assert_allclose(gradient, [2 * (1.0 - target)])
    assert len(traces) == (2 if loss_cls is CallableIndexLoss else 1)


def test_explicit_plan_validates_before_replacement_and_deduplicates_masks():
    loss = CallablePlanLoss(None, None, None)
    losses = LossPreparation(loss)
    masks = [jnp.arange(4) == index for index in (0, 1)]
    calls = []
    original = loss.prepare_selections

    def prepare(indices):
        indices = tuple(indices)
        calls.append(len(indices))
        return original(indices)

    loss.prepare_selections = prepare
    losses.prepare_selections([*masks, masks[0]], (4,))
    assert calls == [2]
    retained = losses.select(jnp.array([0]))
    loss.prepare_selections = lambda indices: ()
    with pytest.raises(ValueError, match="one callable"):
        losses.prepare_selections(masks, (4,))
    losses.select(jnp.array([1]))
    assert losses.select(jnp.array([0])) is retained
    loss.prepare_selections = original
    losses.prepare_selections([], (4,))
    assert losses.prepared_trial_count is None
    assert losses.select(jnp.array([0])) is not retained
