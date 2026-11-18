"""Model nesting, reinforcement targets and semantic support."""
from support import CMR_PARAMS
import numpy as np
from jaxcmr.models.drift_positional_cmr import CMR as DriftCMR
from jaxcmr.models.probe_reinf_positional_cmr import CMR as ProbeCMR
from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR
import importlib
import pytest

from typing import Any
import jax
import jax.numpy as jnp
import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.dataset import make_dataset

_LIST_LENGTH = 4
_RECALLS = jnp.array([[1, 3, 2, 0], [2, 1, 3, 0]], dtype=jnp.int32)
_DATASET: Any = make_dataset(_RECALLS)
_BASE_PARAMS: dict[str, Any] = CMR_PARAMS | {
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
}
_RECENCY_PARAMS: dict[str, Any] = {**_BASE_PARAMS, "primacy_scale": 0.0}
_REPETITION_LIST = [1, 2, 1, 3]


def _run_sequence(model: Any, study_events: list[int], recall_prefix: list[int]) -> Any:
    """Run matched study and recall events on a concrete model instance."""
    for item in study_events:
        model = model.experience(jnp.int32(item))
    model = model.start_retrieving()
    for item in recall_prefix:
        model = model.retrieve(jnp.int32(item))
    return model


def _endpoint_parameters(blend_weight: float) -> dict[str, Any]:
    """Return parameters for endpoint matching tests."""
    return {**_BASE_PARAMS, "mfc_sensitivity": 1.0, "blend_weight": blend_weight}


def test_drift_positional_likelihood_matches_simulation_after_a_repeated_item():
    model = DriftCMR(4, _BASE_PARAMS | {"repetition_drift_rate": 0.7})
    studied = _run_sequence(model, [1, 2, 1, 3], [])
    after_first = studied.retrieve(jnp.int32(1))
    # C is still available at position four after both occurrences of A are removed.
    assert after_first.outcome_probabilities()[3] > 0
    for state in (studied, after_first):
        probabilities = jax.vmap(state.outcome_probability)(jnp.arange(5))
        np.testing.assert_allclose(probabilities, state.outcome_probabilities(), rtol=1e-6)


def test_blend_positional_midpoint_is_finite_and_nontrivial():
    """interior blend weights yield valid non-endpoint behavior."""
    from jaxcmr.models.blend_positional_cmr import CMR as BlendCMR
    from jaxcmr.models.cmr import CMR as StandardCMR
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    midpoint_params = _endpoint_parameters(0.5)
    standard_params = _endpoint_parameters(1.0)
    positional_params = _endpoint_parameters(0.0)
    midpoint = _run_sequence(BlendCMR(_LIST_LENGTH, midpoint_params), _REPETITION_LIST, [2])
    standard = _run_sequence(StandardCMR(_LIST_LENGTH, standard_params), _REPETITION_LIST, [2])
    positional = _run_sequence(
        PositionalCMR(_LIST_LENGTH, positional_params), _REPETITION_LIST, [2]
    )
    midpoint_probs = midpoint.outcome_probabilities()
    assert jnp.all(jnp.isfinite(midpoint_probs)).item()
    assert jnp.isclose(jnp.sum(midpoint_probs), 1.0).item()
    assert not jnp.allclose(midpoint_probs, standard.outcome_probabilities())
    assert not jnp.allclose(midpoint_probs, positional.outcome_probabilities())


def _trace_reinforcement_parameters(reinforcement: float) -> dict[str, Any]:
    """Return parameters for trace-reinforcement model tests."""
    return {**_BASE_PARAMS, "mfc_sensitivity": 1.0, "mcf_first_pres_reinforcement": reinforcement}


def test_probe_reinf_neutral_value_matches_positional_cmr():
    """mcf_first_pres_reinforcement=0 recovers PositionalCMR."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    params = _trace_reinforcement_parameters(0.0)
    positional = _run_sequence(PositionalCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2])
    model = _run_sequence(ProbeCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2])
    assert jnp.allclose(model.outcome_probabilities(), positional.outcome_probabilities())
    assert jnp.allclose(model.context.state, positional.context.state)
    assert jnp.allclose(model.mfc.state, positional.mfc.state)
    assert jnp.allclose(model.mcf.state, positional.mcf.state)


@pytest.mark.parametrize("model_cls", (TraceCMR, ProbeCMR))
def test_trace_and_probe_reinf_do_not_reinforce_unique_items(model_cls):
    """non-repeated lists match PositionalCMR even with reinforcement."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    params = _trace_reinforcement_parameters(5.0)
    study_events = [1, 2, 3, 4]
    positional = _run_sequence(PositionalCMR(_LIST_LENGTH, params), study_events, [])
    model = _run_sequence(model_cls(_LIST_LENGTH, params), study_events, [])
    assert jnp.allclose(model.mcf.state, positional.mcf.state)
    assert jnp.allclose(model.outcome_probabilities(), positional.outcome_probabilities())


def test_probe_reinf_target_first_prior_occurrence():
    """later repetitions reinforce the first prior occurrence only."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    params = _trace_reinforcement_parameters(2.0)
    study_events = [1, 2, 1, 3, 1]
    positional = _run_sequence(PositionalCMR(5, params), study_events, [])
    model = _run_sequence(ProbeCMR(5, params), study_events, [])
    delta = model.mcf.state - positional.mcf.state
    other_columns = delta.at[:, 0].set(0.0)
    assert jnp.any(jnp.abs(delta[:, 0]) > 1e-06).item()
    assert jnp.allclose(other_columns, 0.0)


@pytest.mark.parametrize("post_update", [False, True])
def test_trace_reinf_stores_selected_learning_state(post_update):
    """Store the requested context, with distinct pre- and post-update states."""
    params = _trace_reinforcement_parameters(0.5) | {
        "trace_reinforcement_after_context_update": post_update
    }
    model = TraceCMR(_LIST_LENGTH, params)
    pre_update_state = model.context.state
    model = model.experience(jnp.int32(1))
    expected = model.context.state if post_update else pre_update_state
    assert jnp.allclose(model.trace_contexts[0], expected)
    assert not jnp.allclose(model.context.state, pre_update_state)


def test_probe_reinf_uses_current_mfc_probe_for_first_occurrence():
    """probe variant reinforces the current MFC reconstruction."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR
    from jaxcmr.models.probe_reinf_positional_cmr import CMR as ProbeCMR

    params = _trace_reinforcement_parameters(2.0)
    first_occurrence = jnp.array([True, False, False, False])
    before_repeat = ProbeCMR(_LIST_LENGTH, params)
    before_repeat = before_repeat.experience(jnp.int32(1))
    before_repeat = before_repeat.experience(jnp.int32(2))
    associated_context = before_repeat.mfc.probe(first_occurrence)
    probe = before_repeat.experience(jnp.int32(1))
    positional = PositionalCMR(_LIST_LENGTH, params)
    positional = positional.experience(jnp.int32(1))
    positional = positional.experience(jnp.int32(2))
    positional = positional.experience(jnp.int32(1))
    expected_delta = params["mcf_first_pres_reinforcement"] * jnp.outer(
        associated_context, first_occurrence
    )
    assert jnp.allclose(probe.mcf.state - positional.mcf.state, expected_delta)


def test_additive_semantic_matrix_cmr_zero_scale_matches_base_cmr():
    """matrix-semantic CMR nests base CMR when semantic scale is zero."""
    from jaxcmr.models.cmr import make_factory as make_base_factory
    from jaxcmr.models.additive_semantic_matrix_cmr import make_factory

    base_factory_cls = make_base_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, PositionalTermination
    )
    semantic_factory_cls = make_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, PositionalTermination
    )
    dataset: Any = {
        **_DATASET,
        "pres_itemids": jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=jnp.int32),
    }
    similarity_matrix = jnp.array(
        [[1.0, 0.9, 0.1, 0.1], [0.9, 1.0, 0.1, 0.1], [0.1, 0.1, 1.0, 0.2], [0.1, 0.1, 0.2, 1.0]]
    )
    params = {**_BASE_PARAMS, "semantic_scale": 0.0}
    base_model = _run_sequence(
        base_factory_cls(dataset, None).create_model(params), [1, 2, 3, 4], [1]
    )
    semantic_model = _run_sequence(
        semantic_factory_cls(dataset, similarity_matrix).create_trial_model(jnp.int32(0), params),
        [1, 2, 3, 4],
        [1],
    )
    assert jnp.allclose(base_model.activations(), semantic_model.activations()).item()


def test_additive_semantic_matrix_cmr_boosts_related_candidates():
    """direct similarity support favors related available targets."""
    from jaxcmr.models.additive_semantic_matrix_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, PositionalTermination
    )
    dataset: Any = {
        **_DATASET,
        "pres_itemnos": jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
        "pres_itemids": jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=jnp.int32),
    }
    similarity_matrix = jnp.array(
        [[1.0, 0.9, 0.1, 0.1], [0.9, 1.0, 0.1, 0.1], [0.1, 0.1, 1.0, 0.2], [0.1, 0.1, 0.2, 1.0]]
    )
    zero_params = {**_BASE_PARAMS, "semantic_scale": 0.0}
    semantic_params = {**_BASE_PARAMS, "semantic_scale": 1.0}
    factory = Factory(dataset, similarity_matrix)
    zero_model = _run_sequence(
        factory.create_trial_model(jnp.int32(0), zero_params), [1, 2, 3, 4], [1]
    )
    semantic_model = _run_sequence(
        factory.create_trial_model(jnp.int32(0), semantic_params), [1, 2, 3, 4], [1]
    )
    boosts = semantic_model.activations() - zero_model.activations()
    assert boosts[1] > boosts[3]


    # Swapping the local codes of A and B leaves the same semantic evidence
    # for C after A. The global identities and their similarities stay fixed.
    relabeled = {**dataset, "pres_itemnos": jnp.array([[2, 1, 3, 4], [2, 1, 3, 4]])}
    relabeled_model = _run_sequence(
        Factory(relabeled, similarity_matrix).create_trial_model(jnp.int32(0), semantic_params),
        [2, 1, 3, 4], [2],
    )
    np.testing.assert_allclose(
        relabeled_model.outcome_probabilities()[jnp.array([0, 2, 1, 3, 4])],
        semantic_model.outcome_probabilities(),
        rtol=1e-6,
    )

@pytest.mark.parametrize("weight,module", [(0.0, "positional_cmr"), (1.0, "cmr")])
def test_blend_endpoints_match_before_and_after_retrieval(weight, module):
    from jaxcmr.models.blend_positional_cmr import CMR as Blend

    Reference = importlib.import_module("jaxcmr.models." + module).CMR
    params = _endpoint_parameters(weight)
    reference = _run_sequence(Reference(4, params), _REPETITION_LIST, [])
    blend = _run_sequence(Blend(4, params), _REPETITION_LIST, [])
    np.testing.assert_allclose(blend.outcome_probabilities(), reference.outcome_probabilities())
    np.testing.assert_allclose(blend.context.state, reference.context.state)
    blend, reference = (blend.retrieve(jnp.int32(2)), reference.retrieve(jnp.int32(2)))
    np.testing.assert_allclose(blend.outcome_probabilities(), reference.outcome_probabilities())
    np.testing.assert_allclose(blend.context.state, reference.context.state)


def test_positional_choice_has_correct_gradient_at_sensitivity_one():
    from jaxcmr.models.positional_cmr import CMR

    with jax.enable_x64(True):
        model = CMR(2, _endpoint_parameters(0.0)).replace(
            studied=jnp.array([1, 2]), recallable=jnp.ones(2, dtype=bool),
        )
        model = model.replace(mcf=model.mcf.replace(
            state=jnp.zeros_like(model.mcf.state).at[0].set(jnp.array([2., 1.]))
        ))
        probability = jax.jit(lambda scale: model.replace(mcf_sensitivity=scale).item_probability(0))
        value, derivative = jax.value_and_grad(probability)(1.)
        # For raw support [2, 1], p = (1+e)/(1+2**(-s)+2e).
        # At s=1, dp/ds = (1+e)*log(2)/[2*(1.5+2e)**2].
        epsilon = 2**-23
        np.testing.assert_allclose(value, (1 + epsilon) / (1.5 + 2 * epsilon))
        np.testing.assert_allclose(derivative, (1 + epsilon) * np.log(2) / (2 * (1.5 + 2 * epsilon)**2))
        np.testing.assert_allclose(value, (probability(1 - 1e-5) + probability(1 + 1e-5)) / 2, atol=1e-10)


def test_blended_reinstatement_is_continuous_at_sensitivity_one():
    from jaxcmr.models.blend_positional_cmr import CMR

    with jax.enable_x64(True):
        model = _run_sequence(CMR(4, _endpoint_parameters(0.5)), _REPETITION_LIST, [])
        context_after_recall = jax.jit(lambda scale: model.replace(mfc_sensitivity=scale).retrieve(jnp.int32(1)).context.state)
        before, at_one, after = (context_after_recall(scale) for scale in (1 - 1e-5, 1., 1 + 1e-5))
        np.testing.assert_allclose(at_one, (before + after) / 2, atol=1e-10)
        np.testing.assert_allclose(jax.jacfwd(context_after_recall)(1.), (after - before) / 2e-5, atol=1e-8)


def test_support_ratio_stopping_is_continuous_at_sensitivity_one():
    from jaxcmr.models.ecmr import eCMR
    from jaxcmr.components.termination import SupportRatioTermination

    with jax.enable_x64(True):
        parameters = _BASE_PARAMS | {"choice_sensitivity": 1., "emotion_scale": 0.,
                                    "modulate_emotion_by_primacy": False, "stop_probability_growth": 2.}
        model = eCMR(3, parameters, jnp.zeros(3, dtype=bool),
                     termination_policy_create_fn=SupportRatioTermination)
        model = model.replace(
            studied=jnp.ones(3, dtype=bool), recallable=jnp.array([False, True, True]),
            recall_total=jnp.array(1),
            mcf=model.mcf.replace(state=jnp.zeros_like(model.mcf.state).at[0].set(jnp.array([8., 2., 1.]))),
            emotion_mcf=model.emotion_mcf.replace(state=jnp.zeros_like(model.emotion_mcf.state)),
        )
        stop = jax.jit(lambda scale: model.replace(mcf_sensitivity=scale).stop_probability())
        # Recalled support [8] becomes [1+e]; remaining [2,1] becomes [1+e,1/2+e].
        epsilon = 2**-23
        ratio = (1.5 + 2 * epsilon) / (1 + epsilon)
        np.testing.assert_allclose(stop(1.), 0.05 + np.exp(-2 * ratio))
        np.testing.assert_allclose(stop(1.), (stop(1 - 1e-5) + stop(1 + 1e-5)) / 2, atol=1e-10)
        np.testing.assert_allclose(jax.grad(stop)(1.), np.exp(-2 * ratio) * np.log(2) / (1 + epsilon))
