"""The selected model owns both ordinary events and bulk predictions."""

from support import CMR_PARAMS
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination, PositionalTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
from jaxcmr.loss.transform_sequence_likelihood import (
    ExcludeTerminationLikelihoodLoss,
    MemorySearchLikelihoodLoss,
    mask_trailing_terminations,
)
from jaxcmr.models import optimized_cmr
from jaxcmr.models.cmr import make_factory
from jaxcmr.models.sequence import PreparedPrediction
from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import SequencePredictionFactory

PARAMS = dict(CMR_PARAMS)
REFERENCE = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
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

    def history(factory):
        model = factory.create_model(PARAMS)

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
    expected = jax.jit(lambda: history(reference))()
    actual = jax.jit(lambda: history(optimized))()
    for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("recalls", [(1, 2, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)])
def test_requested_stop_events_match_event_model_and_planning(recalls):
    data = dataset((1, 2, 1), recalls)
    loss = MemorySearchLikelihoodLoss(
        optimized_cmr.ModelFactory, data, None, lambda recalls: jnp.ones_like(recalls, dtype=bool)
    )
    expected = np.asarray(jax.jit(lambda: loss.predict_trial(jnp.array(0), PARAMS))())
    factory = optimized_cmr.ModelFactory(data, None)
    indices = jnp.array([0])
    plans = [
        factory.prepare_predictions(indices),
        factory.prepare_prediction_selections([indices], [jnp.ones((1, 4), dtype=bool)])[0],
    ]
    for plan in plans:
        blocks = jax.jit(lambda bound: bound.probabilities(PARAMS))(plan)
        count = np.count_nonzero(recalls)
        if count:
            np.testing.assert_allclose(blocks[0][0, :count], expected[:count], atol=1e-12)
        np.testing.assert_array_equal(
            np.asarray(blocks[-1])[np.asarray(plan.masks[-1])], expected[count:]
        )
    prepared = loss.prepare_trials(indices)
    value = prepared(PARAMS, (), jnp.empty((0, 1)))[0]
    if sum((v != 0 for v in recalls)) < 2:
        assert np.isinf(value)
    else:
        np.testing.assert_allclose(value, -np.log(expected).sum(), atol=1e-12)


def test_masked_real_recall_still_conditions_next_recall_and_gradient():
    data = dataset()
    mask = lambda recalls: (recalls != 0).at[0].set(False)
    loss = MemorySearchLikelihoodLoss(optimized_cmr.ModelFactory, data, None, mask)
    reference = MemorySearchLikelihoodLoss(REFERENCE, data, None, mask)
    prepared = loss.prepare_trials(jnp.array([0]))
    names = ("learning_rate", "primacy_scale", "recall_drift_rate")
    point = jnp.array([PARAMS[name] for name in names])
    expected = jax.jit(
        jax.value_and_grad(lambda p: reference(jnp.array([0]), PARAMS, names, p[:, None])[0])
    )(point)
    actual = jax.jit(jax.value_and_grad(lambda p: prepared(PARAMS, names, p[:, None])[0]))(point)
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
        prepared = efficient.prepare_trials(indices)
        reference = jax.jit(
            jax.value_and_grad(lambda p: original(indices, PARAMS, names, p[:, None])[0])
        )(point)
        for fn in (
            lambda p: efficient(indices, PARAMS, names, p[:, None])[0],
            lambda p: prepared(PARAMS, names, p[:, None])[0],
        ):
            value, gradient = jax.jit(jax.value_and_grad(fn))(point)
            assert np.isfinite(value) and np.all(np.isfinite(gradient))
            np.testing.assert_allclose(value, reference[0], rtol=1e-05, atol=1e-06)
            np.testing.assert_allclose(gradient, reference[1], rtol=1e-05, atol=1e-06)


def test_changing_shared_model_rule_changes_both_paths(monkeypatch):
    data = dataset()
    factory = optimized_cmr.ModelFactory(data, None)
    prediction = factory.prepare_predictions(jnp.array([0]), data["recalls"] != 0)
    baseline = prediction.probabilities(PARAMS)[0]
    original = optimized_cmr._primacy
    monkeypatch.setattr(optimized_cmr, "_primacy", lambda p, positions: 2 * original(p, positions))
    changed = prediction.probabilities(PARAMS)[0]
    assert not np.allclose(baseline, changed)
    event_loss = MemorySearchLikelihoodLoss(
        optimized_cmr.ModelFactory, data, None, mask_trailing_terminations
    )
    event = jax.jit(lambda: event_loss.predict_trial(jnp.array(0), PARAMS))()
    np.testing.assert_allclose(changed[0], event[:2], atol=1e-12)


def test_compatibility_loss_honors_a_different_model_factory():
    data = dataset()
    positional_stop = make_factory(
        linear_memory.init_mfc, linear_memory.init_mcf, context.init, PositionalTermination
    )
    params = PARAMS | {
        "stop_probability_scale": jnp.array(0.1, dtype=jnp.float64),
        "stop_probability_growth": jnp.array(0.2, dtype=jnp.float64),
    }
    indices = jnp.array([0])
    loss = FastCMRLikelihoodLoss(positional_stop, data, None)
    direct = loss(indices, params, (), jnp.empty((0, 1)))
    prepared = loss.prepare_trials(indices)(params, (), jnp.empty((0, 1)))
    ordinary = ExcludeTerminationLikelihoodLoss(REFERENCE, data, None)(
        indices, PARAMS, (), jnp.empty((0, 1))
    )
    np.testing.assert_allclose(prepared, direct, atol=1e-12)
    assert not np.allclose(prepared, ordinary)


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


def simple_predictions(data, parameters):
    indices, responses = data
    history = jnp.cumsum(responses, axis=1) - responses
    p = parameters["rate"] + 0.05 * indices[:, None] + 0.02 * history
    return (jnp.where(responses == 0, 0.1, 0.9 * jnp.where(responses == 1, p, 1 - p)),)


class BulkFactory(OrdinaryFactory):
    def prepare_predictions(self, indices, mask):
        present, recalls = (self.data["pres_itemnos"][indices], self.data["recalls"][indices])
        responses = jnp.where(recalls != 0, jnp.take_along_axis(present, recalls - 1, axis=1), 0)
        return PreparedPrediction((indices, responses), (mask,), simple_predictions)


def test_generic_preparation_accepts_new_models_and_preserves_trial_identity():
    data = make_dataset(
        jnp.array([[2, 1, 0, 0], [3, 2, 0, 0]]),
        pres_itemnos=jnp.array([[1, 2, 1], [1, 2, 1]]),
        listLength=3,
        subject=1,
    )
    indices = jnp.array([1, 0, 1])
    mask = lambda recalls: (recalls != 0).at[0].set(False)
    assert isinstance(BulkFactory(data, None), SequencePredictionFactory)
    assert not isinstance(OrdinaryFactory(data, None), SequencePredictionFactory)
    results = []
    for factory in (OrdinaryFactory, BulkFactory):
        loss = MemorySearchLikelihoodLoss(factory, data, None, mask)
        for prepared in (loss.prepare_trials(indices), loss.prepare_selections([indices])[0]):
            result = jax.jit(
                jax.value_and_grad(lambda p, bound: bound({}, ("rate",), p[:, None])[0])
            )(jnp.array([0.3]), prepared)
            results.append(result)
    for result in results[1:]:
        np.testing.assert_allclose(result[0], results[0][0], atol=1e-12)
        np.testing.assert_allclose(result[1], results[0][1], atol=1e-12)
