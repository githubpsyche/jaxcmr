"""Mechanistic checks for first-occurrence learning and selection variants."""

from typing import Any
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, lax
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.models import first_bias_positional_cmr, positional_cmr, trace_reinf_positional_cmr

PARAMS: dict[str, Any] = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 0.1,
    "item_support": 0.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "choice_sensitivity": 2.0,
    "mfc_sensitivity": 1.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}
STUDY = [1, 2, 3, 4, 2, 6]


def make_model(module, **params):
    return module.CMR(len(STUDY), PARAMS | params, termination_policy_create_fn=NoStopTermination)


def study(model, events=STUDY):
    for item in events:
        model = model.experience(jnp.int32(item))
    return model


def assert_same_memory(left, right):
    for name in ["context", "mfc", "mcf"]:
        np.testing.assert_allclose(
            getattr(left, name).state, getattr(right, name).state, atol=1e-06
        )


@pytest.mark.parametrize("module", [trace_reinf_positional_cmr, first_bias_positional_cmr])
@pytest.mark.parametrize("serial", [False, True])
def test_zero_mechanisms_match_positional_through_repeated_recall(module, serial):
    """Zero additions preserve baseline study, likelihood, and state updates."""
    base = study(make_model(positional_cmr, allow_repeated_recalls=serial)).start_retrieving()
    model = study(make_model(module, allow_repeated_recalls=serial)).start_retrieving()
    recalls = [1, 2, 3, 4, 2, 6] if serial else [1, 2, 3, 4, 6]
    for item in recalls:
        assert_same_memory(base, model)
        np.testing.assert_allclose(
            base.outcome_probabilities(), model.outcome_probabilities(), atol=1e-06
        )
        base = base.retrieve(jnp.int32(item))
        model = model.retrieve(jnp.int32(item))
    np.testing.assert_array_equal(base.recallable, model.recallable)


@pytest.mark.parametrize("post", [False, True])
def test_reminding_changes_only_earlier_links_and_preserves_encoding_context(post):
    """Earlier reinforcement must not drift the ongoing study context."""
    base = study(make_model(positional_cmr))
    model = study(
        make_model(
            trace_reinf_positional_cmr,
            first_pres_reinforcement=2.0,
            trace_reinforcement_after_context_update=post,
        )
    )
    np.testing.assert_array_equal(base.context.state, model.context.state)
    other_positions = jnp.array([0, 2, 3, 4, 5])
    np.testing.assert_array_equal(base.mfc.state[other_positions], model.mfc.state[other_positions])
    np.testing.assert_array_equal(
        base.mcf.state[:, other_positions], model.mcf.state[:, other_positions]
    )
    assert np.linalg.norm(model.mfc.state[1] - base.mfc.state[1]) > 0.1
    assert np.linalg.norm(model.mcf.state[:, 1] - base.mcf.state[:, 1]) > 0.1


def test_pre_and_post_trace_targets_produce_different_memories():
    """The two target choices must not silently give identical reinforcement."""
    before = study(make_model(trace_reinf_positional_cmr, mcf_first_pres_reinforcement=2.0))
    after = study(
        make_model(
            trace_reinf_positional_cmr,
            mcf_first_pres_reinforcement=2.0,
            trace_reinforcement_after_context_update=True,
        )
    )
    assert np.linalg.norm(before.mcf.state - after.mcf.state) > 0.1
    np.testing.assert_array_equal(before.mfc.state, after.mfc.state)
    np.testing.assert_array_equal(before.context.state, after.context.state)


@pytest.mark.parametrize("direction", ["mcf", "mfc", "both"])
@pytest.mark.parametrize("cost", [0.4, 1.0])
def test_encoding_cost_affects_only_new_occurrence_in_selected_direction(direction, cost):
    """Underencoding removes learned links without erasing the first event."""
    key = (
        "repetition_encoding_cost"
        if direction == "both"
        else f"{direction}_repetition_encoding_cost"
    )
    base = study(make_model(positional_cmr))
    model = study(make_model(trace_reinf_positional_cmr, **{key: cost}))
    prior = study(make_model(positional_cmr), STUDY[:4])
    other_positions = jnp.array([0, 1, 2, 3, 5])
    np.testing.assert_array_equal(base.context.state, model.context.state)
    np.testing.assert_array_equal(base.mfc.state[other_positions], model.mfc.state[other_positions])
    np.testing.assert_array_equal(
        base.mcf.state[:, other_positions], model.mcf.state[:, other_positions]
    )
    for matrix, old, current in [
        ("mfc", prior.mfc.state[4], model.mfc.state[4]),
        ("mcf", prior.mcf.state[:, 4], model.mcf.state[:, 4]),
    ]:
        original = base.mfc.state[4] if matrix == "mfc" else base.mcf.state[:, 4]
        scale = 1 - cost if direction in [matrix, "both"] else 1.0
        np.testing.assert_allclose(current - old, (original - old) * scale, atol=1e-06)


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("fraction", [0.4, 1.0])
def test_allocation_redirects_actual_learning_increments(post, fraction):
    """Allocation respects the current MCF primacy increment, not unit gains."""
    prior = study(
        make_model(
            trace_reinf_positional_cmr,
            repetition_allocation=fraction,
            trace_reinforcement_after_context_update=post,
        ),
        STUDY[:4],
    )
    after = prior.experience(jnp.int32(2))
    earlier = prior.trace_contexts[1]
    current = prior.context.state
    mcf_delta = np.asarray(after.mcf.state - prior.mcf.state)
    mfc_delta = np.asarray(after.mfc.state - prior.mfc.state)
    np.testing.assert_allclose(
        mcf_delta[:, 1], fraction * prior.mcf_learning_rate * earlier, atol=1e-06
    )
    np.testing.assert_allclose(
        mcf_delta[:, 4], (1 - fraction) * prior.mcf_learning_rate * current, atol=1e-06
    )
    np.testing.assert_allclose(
        mfc_delta[1], fraction * prior.mfc_learning_rate * earlier, atol=1e-06
    )
    np.testing.assert_allclose(
        mfc_delta[4], (1 - fraction) * prior.mfc_learning_rate * current, atol=1e-06
    )
    np.testing.assert_allclose(
        np.linalg.norm(mcf_delta, axis=0).sum(), prior.mcf_learning_rate, atol=1e-06
    )
    np.testing.assert_allclose(
        np.linalg.norm(mfc_delta, axis=1).sum(), prior.mfc_learning_rate, atol=1e-06
    )


def test_third_presentation_reinforces_first_not_every_previous_occurrence():
    """The first-target rule remains explicit for lists with >2 occurrences."""
    prior = study(
        make_model(trace_reinf_positional_cmr, mcf_first_pres_reinforcement=2.0), STUDY[:5]
    )
    after = prior.experience(jnp.int32(2))
    np.testing.assert_array_equal(prior.mcf.state[:, 4], after.mcf.state[:, 4])
    assert np.linalg.norm(after.mcf.state[:, 1] - prior.mcf.state[:, 1]) > 0.1


@pytest.mark.parametrize("access", [False, True])
def test_retrieval_bias_keeps_encoding_and_novel_lists_unchanged(access):
    """Retrieval preference is not implemented as covert learning or primacy."""
    base = study(make_model(positional_cmr))
    model = study(make_model(first_bias_positional_cmr, first_pres_bias=3.0, bias_access=access))
    assert_same_memory(base, model)
    np.testing.assert_array_equal(
        model.first_repeated_positions, [False, True, False, False, False, False]
    )
    novel = study(
        make_model(first_bias_positional_cmr, first_pres_bias=3.0, bias_access=access), range(1, 7)
    )
    novel_base = study(make_model(positional_cmr), range(1, 7))
    np.testing.assert_allclose(
        novel.outcome_probabilities(), novel_base.outcome_probabilities(), atol=1e-06
    )


def test_reinstatement_only_bias_does_not_change_access_at_fixed_state():
    """The access ablation must change weights only after choosing the item."""
    base = study(make_model(positional_cmr)).start_retrieving()
    selection = study(make_model(first_bias_positional_cmr, first_pres_bias=3.0)).start_retrieving()
    support = study(
        make_model(first_bias_positional_cmr, first_pres_bias=3.0, bias_access=True)
    ).start_retrieving()
    np.testing.assert_allclose(
        base.outcome_probabilities(), selection.outcome_probabilities(), atol=1e-06
    )
    assert support.item_probability(1) > base.item_probability(1)
    selection_after = selection.retrieve(jnp.int32(2))
    support_after = support.retrieve(jnp.int32(2))
    np.testing.assert_allclose(
        selection_after.context.state, support_after.context.state, atol=1e-06
    )
    assert (
        np.linalg.norm(selection_after.context.state - base.retrieve(jnp.int32(2)).context.state)
        > 0.001
    )
    np.testing.assert_array_equal(
        selection_after.recallable, base.retrieve(jnp.int32(2)).recallable
    )


@pytest.mark.parametrize(
    "module,extra",
    [
        (
            trace_reinf_positional_cmr,
            {"first_pres_reinforcement": 2.0, "repetition_encoding_cost": 0.5},
        ),
        (trace_reinf_positional_cmr, {"repetition_allocation": 0.5}),
        (first_bias_positional_cmr, {"first_pres_bias": 3.0, "bias_access": True}),
    ],
)
@pytest.mark.parametrize("serial", [False, True])
def test_jitted_study_recall_preserves_probability_and_response_contract(module, extra, serial):
    """New parameters work inside the JAX scans used by fitting/simulation."""
    model = make_model(module, **extra, allow_repeated_recalls=serial)
    events = jnp.array(STUDY, dtype=jnp.int32)
    recalls = jnp.array([1, 2, 3, 4, 2, 6] if serial else [1, 2, 3, 4, 6, 0], dtype=jnp.int32)

    @jit
    def run(m):
        m = lax.fori_loop(0, events.size, lambda i, state: state.experience(events[i]), m)
        m = m.start_retrieving()

        def step(state, item):
            probabilities = state.outcome_probabilities()
            return (state.retrieve(item), probabilities)

        return lax.scan(step, m, recalls)

    final, probabilities = run(model)
    assert np.all(np.isfinite(probabilities))
    assert np.all(np.asarray(probabilities) >= 0)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-06)
    if not serial:
        assert not np.asarray(final.recallable).any()


@pytest.mark.parametrize("module", [trace_reinf_positional_cmr, first_bias_positional_cmr])
def test_padding_is_noop_for_learning_and_retrieval_bias_bookkeeping(module):
    """Padding must not invent a first occurrence or consume an encoding slot."""
    model = study(make_model(module), [1, 2, 0, 3])
    expected = study(make_model(module), [1, 2, 3])
    assert_same_memory(model, expected)
    np.testing.assert_array_equal(model.studied, expected.studied)
    assert model.study_index == expected.study_index
