"""Experimental CMR predictions match ordinary event calculations."""

from support import CMR_PARAMS, OPTIMIZED_CMR_PARAMS
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components import context, linear_memory
from jaxcmr.components.response import PositionalStop, RecallUntilExhausted
from jaxcmr.dataset import make_dataset
from jaxcmr.loss.sequence_likelihood import (
    ExcludeTerminationLikelihoodLoss,
    MemorySearchLikelihoodLoss,
    mask_trailing_terminations,
)
from jaxcmr.experimental import optimized_cmr
from jaxcmr.experimental.optimized_cmr import _sequence_likelihood
from jaxcmr.models.cmr import make_factory
from jaxcmr.simulation import simulate_free_recall, simulate_dataset

PARAMS = dict(CMR_PARAMS)
OPTIMIZED_PARAMS = dict(OPTIMIZED_CMR_PARAMS)
REFERENCE = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, response_policy_create_fn=RecallUntilExhausted
)


@pytest.fixture(autouse=True)
def precision():
    with jax.enable_x64(True):
        yield


def dataset(present=(1, 2, 1, 3), recalls=(1, 4, 0, 0)):
    return make_dataset(
        jnp.array([recalls]), pres_itemnos=jnp.array([present]), listLength=len(present), subject=1
    )


@pytest.mark.parametrize(
    "present,recalls",
    [
        ((1, 2, 1, 3), (1, 4, 2, 0)),
        ((1, 2, 3, 4), (4, 1, 0, 0)),
        ((4, 2, 4, 2), (2, 1, 0, 0)),
        ((1, 1, 1, 1), (1, 0, 0, 0)),
    ],
)
def test_event_states_all_outcomes_and_simulation_match_reference(present, recalls):
    data = dataset(present, recalls)

    def history(factory, parameters):
        model = factory.create_model(parameters)

        def study(m, item):
            m = m.experience(item)
            return (m, (m.context.state, m.outcome_probabilities()))

        model, encoded = jax.lax.scan(study, model, data["pres_itemnos"][0])
        model = model.start_retrieving()

        def recall(m, item):
            before = (m.context.state, m.outcome_probabilities())
            return (m.retrieve(item), before)

        labels = jnp.where(
            data["recalls"][0] != 0, data["pres_itemnos"][0][data["recalls"][0] - 1], 0
        )
        _, retrieved = jax.lax.scan(recall, model, labels)
        _, simulated = simulate_free_recall(model, len(present) + 1, jax.random.PRNGKey(57))
        return (encoded, retrieved, simulated)

    reference = REFERENCE(data, None)
    optimized = optimized_cmr.ModelFactory(data, None)
    expected = jax.jit(lambda parameters: history(reference, parameters))(PARAMS)
    actual = jax.jit(lambda parameters: history(optimized, parameters))(OPTIMIZED_PARAMS)
    for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("recalls", [(1, 2, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)])
def test_requested_stop_events_match_event_model_and_planning(recalls):
    data = dataset((1, 2, 1), recalls)
    loss = MemorySearchLikelihoodLoss(
        optimized_cmr.ModelFactory, data, None, lambda recalls: jnp.ones_like(recalls, dtype=bool)
    )
    expected = np.asarray(jax.jit(lambda: loss.predict_trial(jnp.array(0), OPTIMIZED_PARAMS))())
    factory = optimized_cmr.ModelFactory(data, None)
    indices = jnp.array([0])
    plans = [
        factory.prepare_predictions(indices),
        factory.prepare_prediction_selections([indices], [jnp.ones((1, 4), dtype=bool)])[0],
    ]
    for plan in plans:
        blocks = jax.jit(lambda bound: bound.probabilities(OPTIMIZED_PARAMS))(plan)
        count = np.count_nonzero(recalls)
        if count:
            np.testing.assert_allclose(blocks[0][0, :count], expected[:count], atol=1e-12)
        np.testing.assert_array_equal(
            np.asarray(blocks[-1])[np.asarray(plan.masks[-1])], expected[count:]
        )
    prepared = factory.prepare_predictions(indices)
    value = _sequence_likelihood(prepared, OPTIMIZED_PARAMS, (), jnp.empty((0, 1)))[0]
    if sum((v != 0 for v in recalls)) < 2:
        assert np.isinf(value)
    else:
        np.testing.assert_allclose(value, -np.log(expected).sum(), atol=1e-12)


def test_masked_real_recall_still_conditions_next_recall_and_gradient():
    data = dataset()
    mask = lambda recalls: (recalls != 0).at[0].set(False)
    reference = MemorySearchLikelihoodLoss(REFERENCE, data, None, mask)
    prepared = optimized_cmr.ModelFactory(data).prepare_predictions(jnp.array([0]), jax.vmap(mask)(data["recalls"]))
    names = ("learning_rate", "primacy_scale", "recall_drift_rate")
    point = jnp.array([PARAMS[name] for name in names])
    expected = jax.jit(
        jax.value_and_grad(lambda p: reference(jnp.array([0]), PARAMS, names, p[:, None])[0])
    )(point)
    actual = jax.jit(jax.value_and_grad(lambda p: _sequence_likelihood(prepared, OPTIMIZED_PARAMS, names, p[:, None])[0]))(point)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-11)


def test_default_precision_compiles_for_event_and_bulk_gradients():
    with jax.enable_x64(False):
        data = dataset()
        names = ("encoding_drift_rate", "learning_rate", "primacy_scale")
        point = jnp.array([PARAMS[name] for name in names], dtype=jnp.float32)
        indices = jnp.array([0], dtype=jnp.int32)
        original = ExcludeTerminationLikelihoodLoss(REFERENCE, data, None)
        efficient = ExcludeTerminationLikelihoodLoss(optimized_cmr.ModelFactory, data, None)
        prepared = optimized_cmr.ModelFactory(data).prepare_predictions(indices, data["recalls"] != 0)
        reference = jax.jit(
            jax.value_and_grad(lambda p: original(indices, PARAMS, names, p[:, None])[0])
        )(point)
        for fn in (
            lambda p: efficient(indices, OPTIMIZED_PARAMS, names, p[:, None])[0],
            lambda p: _sequence_likelihood(prepared, OPTIMIZED_PARAMS, names, p[:, None])[0],
        ):
            value, gradient = jax.jit(jax.value_and_grad(fn))(point)
            assert np.isfinite(value) and np.all(np.isfinite(gradient))
            np.testing.assert_allclose(value, reference[0], rtol=1e-05, atol=1e-06)
            np.testing.assert_allclose(gradient, reference[1], rtol=1e-05, atol=1e-06)


def test_public_primacy_parameter_changes_event_and_bulk_predictions():
    data = dataset()
    factory = optimized_cmr.ModelFactory(data, None)
    prediction = factory.prepare_predictions(jnp.array([0]), data["recalls"] != 0)
    baseline = prediction.probabilities(OPTIMIZED_PARAMS)[0]
    changed_parameters = OPTIMIZED_PARAMS | {'primacy_scale': 2.0}
    changed = prediction.probabilities(changed_parameters)[0]
    assert not np.allclose(baseline, changed)
    event_loss = MemorySearchLikelihoodLoss(
        optimized_cmr.ModelFactory, data, None, mask_trailing_terminations
    )
    event = jax.jit(lambda: event_loss.predict_trial(jnp.array(0), changed_parameters))()
    np.testing.assert_allclose(changed[0], event[:2], atol=1e-12)


def test_compatibility_loss_honors_a_different_model_factory():
    data = dataset()
    positional_stop = make_factory(
        linear_memory.init_mfc, linear_memory.init_mcf, context.init, response_policy_create_fn=PositionalStop
    )
    params = PARAMS | {
        "stop_probability_scale": jnp.array(0.1, dtype=jnp.float64),
        "stop_probability_growth": jnp.array(0.2, dtype=jnp.float64),
    }
    indices = jnp.array([0])
    loss = ExcludeTerminationLikelihoodLoss(positional_stop, data, None)
    direct = loss(indices, params, (), jnp.empty((0, 1)))
    ordinary = ExcludeTerminationLikelihoodLoss(REFERENCE, data, None)(
        indices, PARAMS, (), jnp.empty((0, 1))
    )
    assert not np.allclose(direct, ordinary)


class SimpleModel(Pytree):
    """A new model with one unrelated parameter and history-dependent choices."""

    def __init__(self, rate):
        self.rate = rate
        self.history = jnp.array(0)
        self.item_count = 2
        self.study_index = jnp.array(0)
        self.recall_total = jnp.array(0)
        self.studied = jnp.ones(2, dtype=bool)
        self.recallable = jnp.ones(2, dtype=bool)
        self.is_active = jnp.array(True)

    def experience(self, choice):
        return self

    def start_retrieving(self):
        return self

    def retrieve(self, choice):
        return self.replace(history=self.history + choice, recall_total=self.recall_total + 1)

    def activations(self):
        p = self.rate + 0.02 * self.history
        return jnp.array([p, 1 - p])

    def outcome_probabilities(self):
        return jnp.concatenate((jnp.array([0.1]), 0.9 * self.activations()))

    def outcome_probability(self, choice):
        p = self.rate + 0.02 * self.history
        return jnp.where(choice == 0, 0.1, 0.9 * jnp.where(choice == 1, p, 1 - p))


class OrdinaryFactory:
    def __init__(self, data, features):
        self.data = data

    def create_model(self, parameters):
        return SimpleModel(parameters["rate"])

    def create_trial_model(self, trial_index, parameters):
        return SimpleModel(parameters["rate"] + 0.05 * trial_index)


def test_ordinary_likelihood_preserves_trial_identity_and_multiplicity():
    data = make_dataset(
        jnp.array([[2, 1, 0, 0], [3, 2, 0, 0]]),
        pres_itemnos=jnp.array([[1, 2, 1], [1, 2, 1]]),
        listLength=3,
        subject=1,
    )
    indices = jnp.array([1, 0, 1])
    mask = lambda recalls: (recalls != 0).at[0].set(False)
    loss = MemorySearchLikelihoodLoss(OrdinaryFactory, data, None, mask)
    evaluate = jax.jit(jax.value_and_grad(
        lambda point, rows: loss(rows, {}, ("rate",), point[:, None])[0]
    ))
    value, gradient = evaluate(jnp.array([0.3]), indices)
    # The selected rows are 1,0,1. After conditioning on the first recall,
    # their scored probabilities are .9*.63, .9*.34, .9*.63.
    expected_loss = -np.log(.9 * .34) - 2 * np.log(.9 * .63)
    expected_gradient = -1 / .34 + 2 / .63
    np.testing.assert_allclose(value, expected_loss, atol=1e-12,
                               err_msg="Selected observations must retain multiplicity")
    np.testing.assert_allclose(gradient, [expected_gradient], atol=1e-12)

    # A/B/A calls pass selections dynamically rather than retaining the last one.
    other, _ = evaluate(jnp.array([0.3]), jnp.array([0, 1, 0]))
    np.testing.assert_allclose(other, -2 * np.log(.9 * .34) - np.log(.9 * .63), atol=1e-12)
    again, _ = evaluate(jnp.array([0.3]), indices)
    np.testing.assert_allclose(again, expected_loss, atol=1e-12)


def test_dataset_simulation_maps_parameter_sets_for_optimized_and_regular_cmr():
    data = dataset((1, 2, 3), (1, 2, 0, 0))
    mask, key = jnp.array([True]), jax.random.PRNGKey(57)

    def simulate(factory, base, drift):
        parameters = {name: jnp.asarray([value]) for name, value in base.items()}
        parameters['subject'] = jnp.array([1])
        return simulate_dataset(
            factory, data, None, {**parameters, 'encoding_drift_rate': drift[None]},
            mask, 2, key,
        )['recalls']

    drift = jnp.array([.3, .6])
    expected = jax.vmap(lambda value: simulate(REFERENCE, PARAMS, value))(drift)
    actual = jax.vmap(lambda value: simulate(optimized_cmr.ModelFactory, OPTIMIZED_PARAMS, value))(drift)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(simulate(optimized_cmr.ModelFactory, OPTIMIZED_PARAMS, drift[0]), expected[0])


@pytest.mark.parametrize("double", [False, True])
def test_prepared_probability_masks_and_gradients(double):
    from jaxcmr.experimental.prepared import PreparedLoss, PreparedPrediction, prediction_likelihood
    with jax.enable_x64(double):
        def probabilities(data, parameters):
            rate = parameters["rate"]
            # The impossible last event must be masked before taking logarithms.
            return (jnp.stack([rate * data, jnp.zeros_like(data)], axis=1),)
        prediction = PreparedPrediction(jnp.array([.5,.75]), (jnp.array([[True,False],[True,False]]),), probabilities)
        loss = PreparedLoss(prediction, prediction_likelihood)
        score = jax.jit(jax.value_and_grad(lambda rate, loss: loss({}, ("rate",), rate.reshape(1,1))[0]))
        value, gradient = score(jnp.array(.4), loss)
        np.testing.assert_allclose(value, -np.log(.2*.3), rtol=1e-6)
        np.testing.assert_allclose(gradient, -2/.4, rtol=1e-6)
        changed = PreparedLoss(PreparedPrediction(jnp.array([.25,.5]), prediction.masks, probabilities), prediction_likelihood)
        np.testing.assert_allclose(score(jnp.array(.4), changed)[0], -np.log(.1*.2), rtol=1e-6)
        np.testing.assert_allclose(score(jnp.array(.4), loss)[0], value)
