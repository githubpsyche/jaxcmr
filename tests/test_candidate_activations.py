"""Tests for candidate-level activation helpers."""

import jax.numpy as jnp

from jaxcmr.components.termination import SupportRatioTermination
from jaxcmr.models.additive_semantic_cmr import CMR as AdditiveSemanticCMR
from jaxcmr.models.additive_semantic_ecmr import AdditiveSemanticECMR
from jaxcmr.models.additive_semantic_retrieval_bias_ecmr import (
    AdditiveSemanticRetrievalBiasECMR,
)
from jaxcmr.models.cmr import CMR
from jaxcmr.models.ecmr import eCMR


_CMR_PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 1.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}

_ECMR_PARAMS = {
    **_CMR_PARAMS,
    "emotion_encoding_drift_rate": 1.0,
    "emotion_recall_drift_rate": 0.6,
    "modulate_emotion_by_primacy": True,
    "emotion_scale": 1.0,
    "phi_emot_modulates_temporal": False,
    "semantic_scale": 0.5,
    "emotional_retrieval_bias": 0.25,
}


def _run_sequence(model):
    for item in [1, 2, 3]:
        model = model.experience(jnp.int32(item))
    model = model.start_retrieving()
    return model.retrieve(jnp.int32(1))


def _assert_activation_wrapper(model):
    assert jnp.allclose(
        model.activations(),
        model.candidate_activations(model.recallable),
    ).item()


def _assert_support_ratio_stop_probability(model):
    model = _run_sequence(model)
    p_stop = model.stop_probability()
    assert jnp.isfinite(p_stop).item()
    assert (p_stop >= 0.0).item()
    assert (p_stop <= 1.0).item()


def test_cmr_activations_delegate_to_candidate_activations():
    """Behavior: base CMR activations are candidate activations for recallables."""
    model = _run_sequence(CMR(3, _CMR_PARAMS))

    _assert_activation_wrapper(model)


def test_additive_semantic_cmr_activations_delegate_to_candidate_activations():
    """Behavior: semantic CMR activations are candidate activations for recallables."""
    connections = jnp.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    model = _run_sequence(
        AdditiveSemanticCMR(3, {**_CMR_PARAMS, "semantic_scale": 0.5}, connections)
    )

    _assert_activation_wrapper(model)


def test_ecmr_activations_delegate_to_candidate_activations():
    """Behavior: eCMR activations are candidate activations for recallables."""
    is_emotional = jnp.array([False, True, False])
    model = _run_sequence(eCMR(3, _ECMR_PARAMS, is_emotional))

    _assert_activation_wrapper(model)


def test_additive_semantic_ecmr_activations_delegate_to_candidate_activations():
    """Behavior: semantic eCMR activations are candidate activations for recallables."""
    is_emotional = jnp.array([False, True, False])
    connections = jnp.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    model = _run_sequence(AdditiveSemanticECMR(3, _ECMR_PARAMS, is_emotional, connections))

    _assert_activation_wrapper(model)


def test_retrieval_bias_ecmr_activations_delegate_to_candidate_activations():
    """Behavior: retrieval-bias eCMR activations are candidate activations."""
    is_emotional = jnp.array([False, True, False])
    connections = jnp.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])
    model = _run_sequence(
        AdditiveSemanticRetrievalBiasECMR(
            3,
            _ECMR_PARAMS,
            is_emotional,
            connections,
        )
    )

    _assert_activation_wrapper(model)


def test_support_ratio_runs_on_candidate_activation_models():
    """Behavior: support-ratio termination runs on candidate activation models."""
    is_emotional = jnp.array([False, True, False])
    connections = jnp.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])

    _assert_support_ratio_stop_probability(
        CMR(3, _CMR_PARAMS, termination_policy_create_fn=SupportRatioTermination)
    )
    _assert_support_ratio_stop_probability(
        AdditiveSemanticCMR(
            3,
            {**_CMR_PARAMS, "semantic_scale": 0.5},
            connections,
            termination_policy_create_fn=SupportRatioTermination,
        )
    )
    _assert_support_ratio_stop_probability(
        eCMR(
            3,
            _ECMR_PARAMS,
            is_emotional,
            termination_policy_create_fn=SupportRatioTermination,
        )
    )
    _assert_support_ratio_stop_probability(
        AdditiveSemanticECMR(
            3,
            _ECMR_PARAMS,
            is_emotional,
            connections,
            termination_policy_create_fn=SupportRatioTermination,
        )
    )
    _assert_support_ratio_stop_probability(
        AdditiveSemanticRetrievalBiasECMR(
            3,
            _ECMR_PARAMS,
            is_emotional,
            connections,
            termination_policy_create_fn=SupportRatioTermination,
        )
    )
