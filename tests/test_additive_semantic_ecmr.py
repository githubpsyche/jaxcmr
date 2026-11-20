"""Semantic support and emotional retrieval bias in additive eCMR variants."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.response import PowerLawStopCompetition, RecallUntilExhausted
from jaxcmr.components.support_ratio import RawSupportRatioRelativeChoice, RawSupportRatioSoftmaxChoice
from jaxcmr.dataset import make_dataset
from jaxcmr.models.additive_semantic_ecmr import (
    make_factory as make_semantic_ecmr_factory,
)
from jaxcmr.models.additive_semantic_retrieval_bias_ecmr import (
    make_factory as make_retrieval_bias_ecmr_factory,
)
from jaxcmr.models.ecmr import make_factory as make_ecmr_factory

_PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "emotion_encoding_drift_rate": 1.0,
    "emotion_recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 1.0,
    "modulate_emotion_by_primacy": True,
    "emotion_scale": 1.0,
    "phi_emot_modulates_temporal": False,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
    "semantic_scale": 0.0,
}


def _dataset(valence=None):
    dataset = make_dataset(jnp.array([[1, 2, 0]], dtype=jnp.int32))
    dataset["pres_itemids"] = dataset["pres_itemnos"]
    dataset["valence"] = jnp.array([[0, -1, 0]], dtype=jnp.int32) if valence is None else valence
    return dataset


def _similarity_matrix():
    return jnp.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])


def _factory(make_factory):
    return make_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, RecallUntilExhausted
    )


def _studied_model(factory_cls, params, features=None, dataset=None):
    factory = factory_cls(_dataset() if dataset is None else dataset, features)
    model = factory.create_trial_model(jnp.int32(0), params)
    for item in [1, 2, 3]:
        model = model.experience(jnp.int32(item))
    return model.start_retrieving()


def test_zero_semantic_scale_matches_base_ecmr():
    """semantic eCMR is base eCMR when semantic scale is zero."""
    base_factory = _factory(make_ecmr_factory)
    semantic_factory = _factory(make_semantic_ecmr_factory)
    params = {**_PARAMS, "semantic_scale": 0.0}
    base_model = _studied_model(base_factory, params).retrieve(jnp.int32(1))
    semantic_model = _studied_model(semantic_factory, params, _similarity_matrix()).retrieve(jnp.int32(1))
    assert jnp.allclose(base_model.outcome_probabilities()[1:], semantic_model.outcome_probabilities()[1:]).item()


def test_power_law_competes_with_absolute_ecmr_evidence():
    params = _PARAMS | {"semantic_scale": 0.7, "stop_log_weight": 0.0, "stop_weight_growth": 2.0}
    base = _studied_model(make_ecmr_factory(response_policy_create_fn=PowerLawStopCompetition), params)
    semantic = _studied_model(
        make_semantic_ecmr_factory(response_policy_create_fn=PowerLawStopCompetition),
        params, _similarity_matrix(),
    )
    assert jnp.allclose(base.item_evidence(), semantic.item_evidence())
    for model in (base, semantic):
        weights = jnp.exp(model.item_evidence())
        assert jnp.allclose(model.outcome_probabilities()[0], 1 / (1 + weights.sum()))
    base, semantic = base.retrieve(1), semantic.retrieve(1)
    evidence = base.item_evidence() + semantic.msem[0]
    assert jnp.allclose(semantic.item_evidence(), evidence)
    weights = jnp.append(4.0, jnp.where(semantic.recallable, jnp.exp(evidence), 0))
    assert jnp.allclose(semantic.outcome_probabilities(), weights / weights.sum())


@pytest.mark.parametrize("policy", [RawSupportRatioSoftmaxChoice, RawSupportRatioRelativeChoice])
def test_raw_ratio_uses_combined_evidence_before_semantic_item_choice(policy):
    params = _PARAMS | {"semantic_scale": 0.7, "choice_sensitivity": 2.0}
    state = _studied_model(
        make_semantic_ecmr_factory(response_policy_create_fn=policy), params, _similarity_matrix()
    ).retrieve(1)
    memory = np.asarray(state.mcf.probe(state.context.state) + state.emotion_mcf.probe(state.emotion_context.state))
    semantic = np.asarray(state.msem[0])
    available = np.asarray(state.recallable)
    evidence = memory + semantic
    support = np.maximum(evidence, 1e-6)
    ratio = support[available].sum() / support[~available].sum()
    stop = params["stop_probability_scale"] + np.exp(-params["stop_probability_growth"] * ratio)
    stop = min(stop, 1.0 - np.finfo(np.float32).eps * available.sum())
    if policy is RawSupportRatioSoftmaxChoice:
        weights = np.exp(evidence)
    else:
        masked_memory_plus_semantic = memory * available + semantic
        weights = (masked_memory_plus_semantic / masked_memory_plus_semantic.max()) ** 2 + np.finfo(np.float32).eps
    weights = weights * available
    expected = np.r_[stop, (1 - stop) * weights / weights.sum()]
    np.testing.assert_allclose(state.outcome_probabilities(), expected, atol=2e-7)


def test_raw_evidence_policy_cannot_silently_drop_retrieval_bias():
    params = _PARAMS | {"emotional_retrieval_bias": 1.0, "stop_log_weight": 0.0, "stop_weight_growth": 2.0}
    model = _studied_model(
        make_retrieval_bias_ecmr_factory(response_policy_create_fn=PowerLawStopCompetition),
        params, _similarity_matrix(),
    )
    with pytest.raises(NotImplementedError, match="candidate_weights"):
        model.outcome_probabilities()


def test_positive_semantic_scale_boosts_related_candidates():
    """semantic support favors items related to the last recall."""
    semantic_factory = _factory(make_semantic_ecmr_factory)
    zero_params = {**_PARAMS, "semantic_scale": 0.0}
    semantic_params = {**_PARAMS, "semantic_scale": 1.0}
    zero_model = _studied_model(semantic_factory, zero_params, _similarity_matrix()).retrieve(jnp.int32(1))
    semantic_model = _studied_model(
        semantic_factory, semantic_params, _similarity_matrix()
    ).retrieve(jnp.int32(1))
    activation_change = semantic_model.outcome_probabilities()[1:] - zero_model.outcome_probabilities()[1:]
    assert activation_change[1] > activation_change[2]


def test_zero_retrieval_bias_matches_semantic_ecmr():
    """retrieval-bias eCMR nests semantic eCMR when bias is zero."""
    semantic_factory = _factory(make_semantic_ecmr_factory)
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    params = {**_PARAMS, "semantic_scale": 0.5, "emotional_retrieval_bias": 0.0}
    semantic_model = _studied_model(semantic_factory, params, _similarity_matrix())
    bias_model = _studied_model(bias_factory, params, _similarity_matrix())
    assert jnp.allclose(semantic_model.outcome_probabilities()[1:], bias_model.outcome_probabilities()[1:]).item()


def test_positive_retrieval_bias_boosts_mixed_emotional_target():
    """positive retrieval bias favors emotional targets in mixed lists."""
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    params = {**_PARAMS, "semantic_scale": 0.5, "emotional_retrieval_bias": 0.0}
    zero_model = _studied_model(bias_factory, params, _similarity_matrix())
    bias_model = _studied_model(
        bias_factory, params | {"emotional_retrieval_bias": 1.0}, _similarity_matrix()
    )
    zero_probs = zero_model.outcome_probabilities()[1:]
    bias_probs = bias_model.outcome_probabilities()[1:]
    assert bias_probs[1] > zero_probs[1]
    assert bias_probs[0] < zero_probs[0]
    assert bias_probs[2] < zero_probs[2]


def test_retrieval_bias_cancels_in_pure_emotional_choice_probabilities():
    """uniform emotional target bias cancels within pure emotional lists."""
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    dataset = _dataset(jnp.array([[-1, -1, -1]], dtype=jnp.int32))
    params = {**_PARAMS, "semantic_scale": 0.5, "emotional_retrieval_bias": 0.0}
    zero_model = _studied_model(bias_factory, params, _similarity_matrix(), dataset=dataset)
    bias_model = _studied_model(
        bias_factory, params | {"emotional_retrieval_bias": 1.0}, _similarity_matrix(), dataset=dataset
    )
    zero_probs = zero_model.outcome_probabilities()[1:]
    bias_probs = bias_model.outcome_probabilities()[1:]
    assert jnp.allclose(zero_probs, bias_probs).item()
