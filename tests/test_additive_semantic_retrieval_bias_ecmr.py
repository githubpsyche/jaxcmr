"""Tests for retrieval-bias semantic eCMR."""

import jax.numpy as jnp
import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.models.additive_semantic_ecmr import make_factory as make_semantic_ecmr_factory
from jaxcmr.models.additive_semantic_retrieval_bias_ecmr import (
    make_factory as make_retrieval_bias_ecmr_factory,
)

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
    "semantic_scale": 0.5,
    "emotional_retrieval_bias": 0.0,
}


def _dataset(valence=None):
    dataset = make_dataset(jnp.array([[1, 2, 0]], dtype=jnp.int32))
    dataset["pres_itemids"] = dataset["pres_itemnos"]
    if valence is None:
        valence = jnp.array([[0, -1, 0]], dtype=jnp.int32)
    dataset["valence"] = valence
    return dataset


def _similarity_matrix():
    return jnp.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])


def _factory(make_factory):
    return make_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, NoStopTermination
    )


def _studied_model(factory_cls, params, dataset=None):
    if dataset is None:
        dataset = _dataset()
    factory = factory_cls(dataset, _similarity_matrix())
    model = factory.create_trial_model(jnp.int32(0), params)
    for item in [1, 2, 3]:
        model = model.experience(jnp.int32(item))
    return model.start_retrieving()


def test_zero_retrieval_bias_matches_semantic_ecmr():
    """retrieval-bias eCMR nests semantic eCMR when bias is zero."""
    semantic_factory = _factory(make_semantic_ecmr_factory)
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    params = {**_PARAMS, "emotional_retrieval_bias": 0.0}
    semantic_model = _studied_model(semantic_factory, params)
    bias_model = _studied_model(bias_factory, params)
    assert jnp.allclose(semantic_model.activations(), bias_model.activations()).item()


def test_positive_retrieval_bias_boosts_mixed_emotional_target():
    """positive retrieval bias favors emotional targets in mixed lists."""
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    zero_model = _studied_model(bias_factory, {**_PARAMS, "emotional_retrieval_bias": 0.0})
    bias_model = _studied_model(bias_factory, {**_PARAMS, "emotional_retrieval_bias": 1.0})
    zero_probs = zero_model.outcome_probabilities()[1:]
    bias_probs = bias_model.outcome_probabilities()[1:]
    assert bias_probs[1] > zero_probs[1]
    assert bias_probs[0] < zero_probs[0]
    assert bias_probs[2] < zero_probs[2]


def test_retrieval_bias_cancels_in_pure_emotional_choice_probabilities():
    """uniform emotional target bias cancels within pure emotional lists."""
    bias_factory = _factory(make_retrieval_bias_ecmr_factory)
    dataset = _dataset(jnp.array([[-1, -1, -1]], dtype=jnp.int32))
    zero_model = _studied_model(
        bias_factory, {**_PARAMS, "emotional_retrieval_bias": 0.0}, dataset=dataset
    )
    bias_model = _studied_model(
        bias_factory, {**_PARAMS, "emotional_retrieval_bias": 1.0}, dataset=dataset
    )
    zero_probs = zero_model.outcome_probabilities()[1:]
    bias_probs = bias_model.outcome_probabilities()[1:]
    assert jnp.allclose(zero_probs, bias_probs).item()
