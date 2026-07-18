"""Tests for additive semantic eCMR."""

import jax.numpy as jnp
from jax import random

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.models.additive_semantic_ecmr import (
    AdditiveSemanticECMR,
    make_factory as make_semantic_ecmr_factory,
)
from jaxcmr.models.ecmr import eCMR, make_factory as make_ecmr_factory
from jaxcmr.simulation import simulate_study_free_recall_and_forced_stop


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
    return jnp.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])


def _factory(make_factory):
    return make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        NoStopTermination,
    )


def _studied_retrieved_model(factory_cls, params, features=None):
    factory = factory_cls(_dataset(), features)
    model = factory.create_trial_model(jnp.int32(0), params)
    for item in [1, 2, 3]:
        model = model.experience(jnp.int32(item))
    model = model.start_retrieving()
    return model.retrieve(jnp.int32(1))


def test_zero_semantic_scale_matches_base_ecmr():
    """Behavior: semantic eCMR is base eCMR when semantic scale is zero.

    Given:
      - Matched base and semantic eCMR factories.
      - A semantic matrix but ``semantic_scale=0``.
    When:
      - Both models encode the list and retrieve the same first item.
    Then:
      - Their next-recall activations match.
    Why this matters:
      - The semantic extension should nest the baseline model.
    """
    # Arrange / Given
    base_factory = _factory(make_ecmr_factory)
    semantic_factory = _factory(make_semantic_ecmr_factory)
    params = {**_PARAMS, "semantic_scale": 0.0}

    # Act / When
    base_model = _studied_retrieved_model(base_factory, params)
    semantic_model = _studied_retrieved_model(
        semantic_factory, params, _similarity_matrix()
    )

    # Assert / Then
    assert jnp.allclose(base_model.activations(), semantic_model.activations()).item()


def test_positive_semantic_scale_boosts_related_candidates():
    """Behavior: semantic support favors items related to the last recall.

    Given:
      - Item 2 is more similar to the first recalled item than item 3.
    When:
      - ``semantic_scale`` is positive.
    Then:
      - The item-2 activation increase is larger than the item-3 increase.
    Why this matters:
      - Additive semantic eCMR should implement semantic clustering pressure.
    """
    # Arrange / Given
    semantic_factory = _factory(make_semantic_ecmr_factory)
    zero_params = {**_PARAMS, "semantic_scale": 0.0}
    semantic_params = {**_PARAMS, "semantic_scale": 1.0}

    # Act / When
    zero_model = _studied_retrieved_model(
        semantic_factory, zero_params, _similarity_matrix()
    )
    semantic_model = _studied_retrieved_model(
        semantic_factory, semantic_params, _similarity_matrix()
    )
    activation_change = semantic_model.activations() - zero_model.activations()

    # Assert / Then
    assert activation_change[1] > activation_change[2]


def test_source_only_and_broad_factories_instantiate_and_simulate():
    """Behavior: semantic eCMR factory supports current emotion-scale variants.

    Given:
      - Source-only, tied broad, temporal-only, and decoupled broad parameters.
    When:
      - A trial model is instantiated and simulated.
    Then:
      - Simulation returns a finite recall sequence with the expected shape.
    Why this matters:
      - One semantic factory must support both current eCMR variants.
    """
    # Arrange / Given
    factory_cls = _factory(make_semantic_ecmr_factory)
    dataset = _dataset()

    # Act / When / Assert / Then
    variant_params = [
        {"phi_emot_modulates_temporal": False},
        {
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        {
            "emotion_scale": 0.0,
            "temporal_emotion_scale": 1.0,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
        {
            "emotion_scale": 0.5,
            "temporal_emotion_scale": 1.0,
            "phi_emot_modulates_temporal": True,
            "modulate_temporal_emotion_by_primacy": True,
        },
    ]
    for variant in variant_params:
        params = {
            **_PARAMS,
            "semantic_scale": 0.5,
            **variant,
        }
        model = factory_cls(dataset, _similarity_matrix()).create_trial_model(
            jnp.int32(0), params
        )
        assert isinstance(model, AdditiveSemanticECMR)
        assert isinstance(model, eCMR)

        _, recalls = simulate_study_free_recall_and_forced_stop(
            model,
            dataset["pres_itemnos"][0],
            dataset["recalls"][0],
            random.PRNGKey(0),
        )
        assert recalls.shape == dataset["recalls"][0].shape
        assert jnp.all(jnp.isfinite(recalls)).item()
