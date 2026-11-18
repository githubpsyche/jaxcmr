"""Conditional first-recall scoring and count-controlled simulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from support import CMR_PARAMS, OPTIMIZED_CMR_PARAMS

from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.dataset import make_dataset
from jaxcmr.loss.sequence_likelihood import (
    ExcludeFirstRecallAndTerminationLikelihoodLoss,
    ExcludeTerminationLikelihoodLoss,
    mask_first_recall_and_terminations,
)
from jaxcmr.models import additive_semantic_matrix_cmr, cmr
from jaxcmr.experimental import optimized_cmr
from jaxcmr.experimental.optimized_cmr import _sequence_likelihood
from jaxcmr.simulation import (
    simulate_dataset,
    simulate_study_first_recall_and_forced_stop,
)


def factory_for(module):
    return module.make_factory(
        linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
    )


@pytest.fixture(autouse=True)
def precision():
    with jax.enable_x64(True):
        yield


def dataset(recalls=(2, 4, 1, 0, 0)):
    data = make_dataset(
        jnp.array([recalls]), pres_itemnos=jnp.array([[3, 1, 4, 2]]),
        listLength=4, subject=7,
    )
    data['pres_itemids'] = jnp.array([[8, 5, 7, 6]])
    return data


@pytest.mark.parametrize('recalls', [(2, 4, 1, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, 0)])
@pytest.mark.parametrize('factory,parameters', [
    (factory_for(cmr), CMR_PARAMS), (optimized_cmr.ModelFactory, OPTIMIZED_CMR_PARAMS),
])
def test_conditional_product_and_prepared_gradients(recalls, factory, parameters):
    data = dataset(recalls)
    params = dict(parameters)
    loss = ExcludeFirstRecallAndTerminationLikelihoodLoss(factory, data, None)
    model = factory(data, None).create_trial_model(jnp.int32(0), params)
    for item in data['pres_itemnos'][0]:
        model = model.experience(item)
    model = model.start_retrieving()
    expected = 0.0
    first_probability = None
    for index, position in enumerate(recalls):
        if not position:
            break
        item = data['pres_itemnos'][0, position - 1]
        probability = model.outcome_probability(item)
        if index:
            expected -= np.log(float(probability))
        else:
            first_probability = float(probability)
        model = model.retrieve(item)
    rows = jnp.array([0])
    names = ('choice_sensitivity', 'learning_rate')
    point = jnp.array([params[key] for key in names])
    direct = jax.jit(jax.value_and_grad(lambda p: loss(rows, params, names, p[:, None])[0]))(point)
    np.testing.assert_allclose(direct[0], expected, atol=1e-11)
    if factory is optimized_cmr.ModelFactory:
        optimized = factory(data)
        scoring = jax.vmap(mask_first_recall_and_terminations)(data["recalls"])
        for sequence in (
            optimized.prepare_predictions(rows, scoring),
            optimized.prepare_prediction_selections([rows], [scoring])[0],
        ):
            actual = jax.jit(jax.value_and_grad(
                lambda point: _sequence_likelihood(sequence, params, names, point[:, None])[0]
            ))(point)
            np.testing.assert_allclose(actual[0], direct[0], atol=1e-11)
            np.testing.assert_allclose(actual[1], direct[1], atol=1e-10)
    if first_probability is not None:
        original = ExcludeTerminationLikelihoodLoss(factory, data, None)
        full = original(rows, params, names, point[:, None])[0]
        np.testing.assert_allclose(full - direct[0], -np.log(first_probability), atol=1e-11)
    if sum(x > 0 for x in recalls) <= 1:
        np.testing.assert_array_equal(direct[1], 0)


def test_combined_mask_scores_only_recalls_after_the_first():
    recalls = jnp.array([2, 4, 1, 0, 0])
    np.testing.assert_array_equal(mask_first_recall_and_terminations(recalls), [False, True, True, False, False])


@pytest.mark.parametrize('count', [0, 1, 2, 4])
@pytest.mark.parametrize('semantic', [False, True])
def test_simulation_forces_start_count_and_exact_retrieval_state(count, semantic):
    observed = (2, 4, 1, 3)[:count] + (0,) * (5 - count)
    data = dataset(observed)
    factory = factory_for(additive_semantic_matrix_cmr if semantic else cmr)
    features = np.eye(8) + 0.2 if semantic else None
    params = {**CMR_PARAMS, 'semantic_scale': 2.0}
    model = factory(data, features).create_trial_model(jnp.int32(0), params)
    present = data['pres_itemnos'][0]
    trial = jnp.where(data['recalls'][0] > 0, present[data['recalls'][0] - 1], 0)
    def simulate(key):
        return simulate_study_first_recall_and_forced_stop(model, present, trial, key)
    keys = jax.random.split(jax.random.PRNGKey(19), 8)
    states, recalls = jax.jit(jax.vmap(simulate))(keys)
    np.testing.assert_array_equal(np.count_nonzero(recalls, axis=1), count)
    np.testing.assert_array_equal(recalls[:, count:], 0)
    np.testing.assert_array_equal(recalls[:, 0], trial[0])
    assert all(len(set(row[row > 0])) == count for row in np.asarray(recalls))
    np.testing.assert_array_equal(states.recall_total, count)
    # Independently replay returned choices: all state fields must match.
    state, result = jax.jit(simulate)(keys[0])
    reference = model
    for item in present:
        reference = reference.experience(item)
    reference = reference.start_retrieving()
    for item in result[:count]:
        reference = reference.retrieve(item)
    for actual, expected in zip(jax.tree.leaves(state), jax.tree.leaves(reference), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    # Dataset API converts local item labels back to actual study positions.
    subject_parameters = {k: jnp.array([v]) for k, v in params.items()}
    subject_parameters['subject'] = jnp.array([7])
    simulated = simulate_dataset(
        factory, data, features, subject_parameters,
        jnp.array([True]), 8, jax.random.PRNGKey(19),
        simulate_trial_fn=simulate_study_first_recall_and_forced_stop,
    )
    np.testing.assert_array_equal(simulated['recalls'][:, 0], observed[0])
    np.testing.assert_array_equal(np.count_nonzero(simulated['recalls'], axis=1), count)
