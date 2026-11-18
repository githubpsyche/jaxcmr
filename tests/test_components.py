"""Numerical boundaries of context drift, memory updates, and stopping."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.components import context, instance_memory, linear_memory
from jaxcmr.components.termination import (
    NoStopTermination,
    PositionalTermination,
    RetrievalDependentTermination,
    SupportRatioTermination,
    positional_stop_probability,
    stop_when_exhausted,
    support_ratio_stop_probability,
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
        (PositionalTermination, [True] * 3, True, 5, 0.01 * np.exp(1.0)),
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
    inputs = {"is_active": active, "available_count": sum(recallable)}
    if policy is NoStopTermination:
        assert jax.jit(stop_when_exhausted)(**inputs) == result
    elif policy is PositionalTermination:
        schedule = 0.01 * jnp.exp(0.5 * jnp.arange(3))
        assert jax.jit(positional_stop_probability)(schedule, total, **inputs) == result
    elif policy is SupportRatioTermination:
        support = np.array([2.0, 4.0, 6.0])
        assert jax.jit(support_ratio_stop_probability)(
            support[recallable].sum(), support[~np.array(recallable)].sum(),
            scale=0.01, growth=0.5, **inputs,
        ) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "policy",
    [
        PositionalTermination,
        RetrievalDependentTermination,
        SupportRatioTermination,
    ],
)
@pytest.mark.parametrize("dtype", [None, jnp.float32], ids=["default", "float32"])
def test_stopping_preserves_precision_and_fixed_probability_margin(policy, dtype):
    parameters = {
        "stop_probability_scale": jnp.asarray(1.0, dtype=dtype),
        "stop_probability_growth": jnp.asarray(0.5, dtype=dtype),
    }
    termination = policy(3, parameters)
    support = jnp.array([2.0, 4.0, 6.0], dtype=dtype)
    expected_dtype = support.dtype

    def probability(recallable, active):
        model = SimpleNamespace(
            item_count=3,
            outcome_probability=lambda item: jnp.asarray(1.0, dtype=dtype),
            recallable=recallable,
            is_active=active,
            recall_total=jnp.asarray(0),
            studied=jnp.ones(3, dtype=bool),
            candidate_activations=lambda mask: support * mask,
        )
        return termination.stop_probability(model)

    compiled = jax.jit(probability)
    # FLOAT32_EPS remains exactly 2**-23, even in float64. Probability one reaches
    # the existing ceiling; exhausted and inactive models must instead stop.
    for recallable, active, expected in [
        ([False, True, True], True, 1.0 - 2 * 2**-23),
        ([True, True, True], True, 1.0 - 3 * 2**-23),
        ([False, True, True], False, 1.0),
        ([False, False, False], True, 1.0),
    ]:
        for evaluate in (probability, compiled):
            result = evaluate(jnp.array(recallable), jnp.asarray(active))
            assert result.dtype == expected_dtype
            np.testing.assert_array_equal(result, expected)


def test_instance_memory_stores_traces_and_distinguishes_scaled_and_raw_probes():
    memory = instance_memory.InstanceMemory(
        jnp.zeros((3, 5)), jnp.zeros(5), 0, 2.0, 2, 3
    )
    changed = memory.associate(jnp.array([1.0, 0]), jnp.array([2.0, 4, 1]), 0.5)
    changed = changed.associate(jnp.array([0.0, 1]), jnp.array([1.0, 0, 2]), 0.25)
    np.testing.assert_array_equal(memory.state, np.zeros((3, 5)))
    np.testing.assert_array_equal(
        changed.state,
        [
            [1, 0, 1, 2, 0.5],
            [0, 1, 0.25, 0, 0.5],
            [0, 0, 0, 0, 0],
        ],
    )
    assert changed.study_index == 2
    cue = jnp.array([2.0, 1])
    # Raw supports are [2, 1, 0]. Normalized powers at two are [1, 1/4, 0].
    np.testing.assert_allclose(
        jax.jit(lambda mem: mem.probe(cue))(changed), [1.0625, 2, 0.625]
    )
    np.testing.assert_allclose(
        jax.jit(lambda mem: mem.probe_without_scale(cue))(changed), [2.25, 4, 1.5]
    )
    np.testing.assert_array_equal(changed.probe(jnp.zeros(2)), np.zeros(3))

    def total_support(sensitivity):
        return changed.replace(trace_activation_scale=sensitivity).probe(cue).sum()

    # Sum of learned outputs: 3.5 + 0.75 * (1/2)**sensitivity.
    sensitivities = jnp.array([0.999, 1.0, 1.001, 2.0])
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(total_support)))(
        sensitivities
    )
    np.testing.assert_allclose(
        values, 3.5 + 0.75 * 0.5 ** np.asarray(sensitivities), rtol=1e-6
    )
    np.testing.assert_allclose(
        derivatives, 0.75 * 0.5 ** np.asarray(sensitivities) * np.log(0.5), rtol=1e-6
    )


@pytest.mark.parametrize("memory_module", [linear_memory, instance_memory], ids=["linear", "instance"])
@pytest.mark.parametrize(
    "direction,expected",
    [
        ("mfc", [[0, 0.6, 0, 0, 0], [0, 0, 0.6, 0, 0]]),
        ("mcf", [[0, 0], [1.5, 0.2], [0.2, 1.5], [0, 0], [0, 0]]),
    ],
)
def test_memory_creators_align_items_in_expanded_context(
    memory_module, direction, expected
):
    parameters = {
        "learning_rate": 0.4,
        "item_support": 1.5,
        "shared_support": 0.2,
        "mcf_trace_sensitivity": 1.0,
    }
    creator = (
        memory_module.init_mfc if direction == "mfc" else memory_module.init_mcf
    )
    memory = creator(2, parameters, context.TemporalContext.init_expanded(2))
    if memory_module is instance_memory:
        np.testing.assert_array_equal(memory.state[2:], np.zeros((2, 7)))
    # Basis probes expose item/context alignment, including start and outlist units.
    responses = jax.jit(jax.vmap(memory.probe))(jnp.eye(memory.input_size))
    np.testing.assert_allclose(responses, expected, rtol=1e-6)


def test_instance_memory_composes_with_ordinary_cmr_and_differentiates_at_one():
    from support import CMR_PARAMS
    from jaxcmr.dataset import make_dataset
    from jaxcmr.models.cmr import make_factory

    data = make_dataset([1, 2, 0], pres_itemnos=[1, 2, 1, 3])
    parameters = CMR_PARAMS | {
        "mfc_trace_sensitivity": 1.0,
        "stop_probability_scale": 0.05,
        "stop_probability_growth": 0.2,
    }
    instance_factory = make_factory(
        instance_memory.init_mfc,
        instance_memory.init_mcf,
        context.init,
        PositionalTermination,
    )(data, None)
    linear_factory = make_factory(
        linear_memory.init_mfc,
        linear_memory.init_mcf,
        context.init,
        PositionalTermination,
    )(data, None)

    def outcome_probabilities(factory, sensitivity):
        model = factory.create_trial_model(
            jnp.int32(0),
            parameters
            | {
                "mcf_trace_sensitivity": sensitivity,
            },
        )
        for item in (1, 2, 1, 3):
            model = model.experience(jnp.int32(item))
        model = model.start_retrieving()
        return jnp.stack(
            [
                model.outcome_probabilities(),
                model.retrieve(jnp.int32(1)).outcome_probabilities(),
            ]
        )

    # At one, trace and matrix probes differ only by a common positive
    # factor; ordinary CMR normalizes it away before choosing an item.
    probabilities = jax.jit(
        lambda scale: outcome_probabilities(instance_factory, scale)
    )
    np.testing.assert_allclose(
        probabilities(1.0), outcome_probabilities(linear_factory, 1.0), atol=1e-6
    )
    batch = jax.vmap(probabilities)(jnp.array([0.7, 1.0, 1.6]))
    np.testing.assert_allclose(batch.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(np.asarray(batch) >= 0)
    np.testing.assert_array_equal(batch[:, 1, 1], np.zeros(3))
    np.testing.assert_array_equal(batch[:, :, 4], np.zeros((3, 2)))
    assert not np.allclose(batch[0], batch[2])
    derivative = jax.grad(lambda scale: probabilities(scale)[0, 2])(1.0)
    finite_difference = (
        probabilities(1.001)[0, 2] - probabilities(0.999)[0, 2]
    ) / 0.002
    np.testing.assert_allclose(derivative, finite_difference, rtol=5e-3, atol=1e-6)
