"""Tests for additive semantic eCMR."""

import jax.numpy as jnp
import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.models.additive_semantic_ecmr import make_factory as make_semantic_ecmr_factory
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


def _dataset():
    dataset = make_dataset(jnp.array([[1, 2, 0]], dtype=jnp.int32))
    dataset["pres_itemids"] = dataset["pres_itemnos"]
    dataset["valence"] = jnp.array([[0, -1, 0]], dtype=jnp.int32)
    return dataset


def _similarity_matrix():
    return jnp.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])


def _factory(make_factory):
    return make_factory(
        LinearMemory.init_mfc, LinearMemory.init_mcf, TemporalContext.init, NoStopTermination
    )


def _studied_retrieved_model(factory_cls, params, features=None):
    factory = factory_cls(_dataset(), features)
    model = factory.create_trial_model(jnp.int32(0), params)
    for item in [1, 2, 3]:
        model = model.experience(jnp.int32(item))
    model = model.start_retrieving()
    return model.retrieve(jnp.int32(1))


def test_zero_semantic_scale_matches_base_ecmr():
    """semantic eCMR is base eCMR when semantic scale is zero."""
    base_factory = _factory(make_ecmr_factory)
    semantic_factory = _factory(make_semantic_ecmr_factory)
    params = {**_PARAMS, "semantic_scale": 0.0}
    base_model = _studied_retrieved_model(base_factory, params)
    semantic_model = _studied_retrieved_model(semantic_factory, params, _similarity_matrix())
    assert jnp.allclose(base_model.activations(), semantic_model.activations()).item()


def test_positive_semantic_scale_boosts_related_candidates():
    """semantic support favors items related to the last recall."""
    semantic_factory = _factory(make_semantic_ecmr_factory)
    zero_params = {**_PARAMS, "semantic_scale": 0.0}
    semantic_params = {**_PARAMS, "semantic_scale": 1.0}
    zero_model = _studied_retrieved_model(semantic_factory, zero_params, _similarity_matrix())
    semantic_model = _studied_retrieved_model(
        semantic_factory, semantic_params, _similarity_matrix()
    )
    activation_change = semantic_model.activations() - zero_model.activations()
    assert activation_change[1] > activation_change[2]
