"""Numerical boundaries of context drift, memory updates, and stopping."""

from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import (
    NoStopTermination,
    RetrievalDependentTermination,
    PositionalTermination,
    SupportRatioTermination,
)


@pytest.mark.parametrize(
    "state,incoming,rate,expected",
    [
        ([0.6, 0.8], [0, 0], 0.6, [0.48, 0.64]),
        ([0.6, 0.8], [0.6, 0.8], 0.5, [0.6, 0.8]),
        ([0.6, 0.8, 0], [0.8, 0, 0.6], 0.4, [0.7666461, 0.5955281, 0.24]),
    ],
)
def test_context_update_known_vectors(state, incoming, rate, expected):
    model = context.init(len(state) - 1).replace(state=jnp.array(state))
    np.testing.assert_allclose(
        model.integrate(jnp.array(incoming), rate).state, expected, atol=1e-7
    )


def test_context_small_inputs_and_outlist_overlap():
    model = context.init(40)
    norms = []
    for index in range(40):
        model = model.integrate(jnp.zeros(41).at[index + 1].set(0.001), 0.999)
        norms.append(jnp.linalg.norm(model.state))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    expanded = context.TemporalContext.init_expanded(2).replace(
        state=jnp.array([0.6, 0.8, 0, 0, 0])
    )
    np.testing.assert_array_equal(expanded.outlist_input, [0, 0, 0, 1, 0])
    updated = expanded.integrate_with_outlist(
        jnp.array([0.0, 1, 0, 0, 0]), ratio=0.6, drift_rate=0.4
    )
    np.testing.assert_allclose(updated.state, [0.4128904, 0.8833407, 0, 0.2218801, 0], atol=1e-7)
    assert updated.next_outlist_unit == expanded.next_outlist_unit + 1


def test_linear_memory_association_probe_and_erasure():
    memory = linear_memory.LinearMemory(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    changed = memory.associate(jnp.array([1.0, 0]), jnp.array([0.5, 0.5]), 0.3)
    np.testing.assert_allclose(changed.state, [[1.15, 2.15], [3, 4]])
    np.testing.assert_allclose(changed.probe(jnp.array([1.0, 0])), [1.15, 2.15])
    np.testing.assert_array_equal(changed.zero_out(1).state[1], [0, 0])
    np.testing.assert_array_equal(memory.state, [[1, 2], [3, 4]])


@pytest.mark.parametrize(
    "policy,recallable,active,total,expected",
    [
        (RetrievalDependentTermination, [True] * 3, True, 0, 0.1),
        (NoStopTermination, [True, True, False], True, 0, 0.0),
        (NoStopTermination, [False, False, False], True, 0, 1.0),
        (PositionalTermination, [True] * 3, True, 0, 0.01),
        (PositionalTermination, [True] * 3, True, 2, 0.01 * np.exp(1.0)),
        (PositionalTermination, [True] * 3, False, 0, 1.0),
        (SupportRatioTermination, [False] * 3, True, 0, 1.0),
        (SupportRatioTermination, [False, True, True], True, 1, 0.01 + np.exp(-2.5)),
    ],
)
def test_stopping_probabilities(policy, recallable, active, total, expected):
    model = SimpleNamespace(
        item_count=3,
        outcome_probability=lambda item: jnp.where(item == 3, 0.1, 0.0),
        recallable=jnp.array(recallable),
        is_active=jnp.array(active),
        recall_total=jnp.array(total),
        studied=jnp.ones(3, dtype=bool),
        candidate_activations=lambda mask: jnp.array([2.0, 4.0, 6.0]) * mask,
    )
    result = policy(
        3, {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    ).stop_probability(model)
    assert result == pytest.approx(expected, rel=1e-6)
