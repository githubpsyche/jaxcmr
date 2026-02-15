"""Tests for all non-base CMR model variants.

Each test constructs a factory, creates a model, runs a study-recall
cycle, and verifies behavioral properties: finite activations and
temporal recency (last-studied item has highest activation).
"""

from typing import Any

import jax.numpy as jnp
import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.helpers import make_dataset

# ── shared dataset and base parameters ──────────────────────────────────────

_LIST_LENGTH = 4

_RECALLS = jnp.array([[1, 3, 2, 0], [2, 1, 3, 0]], dtype=jnp.int32)

_DATASET: Any = make_dataset(_RECALLS)

_BASE_PARAMS: dict[str, Any] = {
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
    "choice_sensitivity": 2.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}


def _run_study_recall_cycle(factory_cls: Any, params: dict, dataset: Any = _DATASET) -> Any:
    """Construct a factory, create a model, study items, then return activations."""
    factory = factory_cls(dataset, None)
    model = factory.create_model(params)
    for i in range(1, _LIST_LENGTH + 1):
        model = model.experience(jnp.int32(i))
    model = model.start_retrieving()
    return model.activations()


def _run_semantic_study_recall_cycle(
    factory_cls: Any, params: dict, dataset: Any, features: Any
) -> Any:
    """Same as _run_study_recall_cycle but for semantic models that need features."""
    factory = factory_cls(dataset, features)
    model = factory.create_trial_model(jnp.int32(0), params)
    for i in range(1, _LIST_LENGTH + 1):
        model = model.experience(jnp.int32(i))
    model = model.start_retrieving()
    return model.activations()


# ── positional CMR ──────────────────────────────────────────────────────────

def test_positional_cmr_shows_recency():
    """Behavior: Positional CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run with mfc_sensitivity=1.0.
    Then:
      - Activations are finite with correct shape.
      - Last item has higher activation than first (recency).
      - Activations are monotonically increasing (full recency gradient).
    Why this matters:
      - Verifies that the positional representation variant preserves
        the temporal recency gradient from sequential encoding.
    """
    # Arrange / Given
    from jaxcmr.models.positional_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    params = {**_BASE_PARAMS, "mfc_sensitivity": 1.0}

    # Act / When
    activations = _run_study_recall_cycle(Factory, params)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations.shape == (_LIST_LENGTH,)
    assert activations[-1] > activations[0]  # recency: item 4 > item 1
    # Full recency gradient: each item more active than the previous
    for i in range(_LIST_LENGTH - 1):
        assert activations[i + 1] > activations[i]


# ── distinct contexts CMR ───────────────────────────────────────────────────

def test_distinct_contexts_cmr_shows_recency():
    """Behavior: Distinct Contexts CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run.
    Then:
      - Activations are finite.
      - Last item has higher activation than first (recency).
    Why this matters:
      - Verifies that distinct per-item contexts preserve the recency
        gradient from encoding-phase temporal context drift.
    """
    # Arrange / Given
    from jaxcmr.models.distinct_contexts_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )

    # Act / When
    activations = _run_study_recall_cycle(Factory, _BASE_PARAMS)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations[-1] > activations[0]  # recency


# ── no reinstate CMR ────────────────────────────────────────────────────────

def test_no_reinstate_cmr_shows_recency():
    """Behavior: No-Reinstate CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run.
    Then:
      - Activations are finite.
      - Last item has higher activation than first (recency).
    Why this matters:
      - Even without context reinstatement at recall, encoding-phase
        recency should be preserved in the initial activations.
    """
    # Arrange / Given
    from jaxcmr.models.no_reinstate_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )

    # Act / When
    activations = _run_study_recall_cycle(Factory, _BASE_PARAMS)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations[-1] > activations[0]  # recency


# ── blend positional CMR ────────────────────────────────────────────────────

def test_blend_positional_cmr_shows_recency():
    """Behavior: Blend Positional CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run with mfc_sensitivity=1.0 and
        blend_weight=0.5.
    Then:
      - Activations are finite with recency.
      - Full recency gradient: each successive item is more active.
    Why this matters:
      - Verifies the dual-stream blend variant preserves temporal
        recency from the encoding phase.
    """
    # Arrange / Given
    from jaxcmr.models.blend_positional_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    params = {**_BASE_PARAMS, "mfc_sensitivity": 1.0, "blend_weight": 0.5}

    # Act / When
    activations = _run_study_recall_cycle(Factory, params)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations[-1] > activations[0]  # recency
    for i in range(_LIST_LENGTH - 1):
        assert activations[i + 1] > activations[i]


# ── drift positional CMR ────────────────────────────────────────────────────

def test_drift_positional_cmr_shows_recency():
    """Behavior: Drift Positional CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run.
    Then:
      - Activations are finite.
      - Last item has higher activation than first (recency).
    Why this matters:
      - Verifies the per-repetition drift variant preserves the
        standard temporal recency gradient.
    """
    # Arrange / Given
    from jaxcmr.models.drift_positional_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    params = {**_BASE_PARAMS, "repetition_drift_rate": 0.3}

    # Act / When
    activations = _run_study_recall_cycle(Factory, params)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations[-1] > activations[0]  # recency


# ── reinforcement positional CMR ────────────────────────────────────────────

def test_reinf_positional_cmr_shows_recency():
    """Behavior: Reinforcement Positional CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run with first_pres_reinforcement=0.5.
    Then:
      - Activations are finite with recency.
    Why this matters:
      - Verifies that the first-presentation reinforcement variant
        preserves the temporal recency gradient.
    """
    # Arrange / Given
    from jaxcmr.models.reinf_positional_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    params = {
        **_BASE_PARAMS,
        "mfc_sensitivity": 1.0,
        "first_pres_reinforcement": 0.5,
    }

    # Act / When
    activations = _run_study_recall_cycle(Factory, params)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations[-1] > activations[0]  # recency


# ── dual cue CMR ────────────────────────────────────────────────────────────

def test_dual_cue_cmr_nonuniform_activations():
    """Behavior: Dual Cue CMR produces a non-uniform activation gradient.

    Given:
      - Items [1, 2, 3, 4] studied in order.
    When:
      - A study-recall cycle is run with global_mcf_sensitivity=2.0.
    Then:
      - Activations are finite with correct shape.
      - Activations are non-uniform (items have different support).
      - First item has highest activation (primacy from the dual-cue
        global MCF mechanism dominates over recency).
    Why this matters:
      - Verifies the dual-cue variant produces differential item support
        and that the global MCF introduces a primacy-dominant pattern.
    """
    # Arrange / Given
    from jaxcmr.models.dual_cue_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
    )
    params = {**_BASE_PARAMS, "global_mcf_sensitivity": 2.0}

    # Act / When
    factory = Factory(_DATASET, None)
    model = factory.create_model(params)
    for i in range(1, _LIST_LENGTH + 1):
        model = model.experience(jnp.int32(i))
    model = model.start_retrieving()
    activations = model.activations()

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations.shape == (_LIST_LENGTH,)
    assert not jnp.allclose(activations, activations[0])  # non-uniform
    assert activations[0] > activations[-1]  # primacy dominates


# ── additive semantic CMR ───────────────────────────────────────────────────

def test_additive_semantic_cmr_shows_recency():
    """Behavior: Additive Semantic CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
      - Non-identity feature matrix with inter-item similarity.
    When:
      - A study-recall cycle is run with semantic_scale=0.5.
    Then:
      - Activations are finite with correct shape and recency.
    Why this matters:
      - Verifies the additive semantic variant (which adds pre-experimental
        semantic associations to MCF) preserves temporal recency.
    """
    # Arrange / Given
    from jaxcmr.models.additive_semantic_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    dataset: Any = {
        **_DATASET,
        "pres_itemids": jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32),
    }
    features = jnp.array([
        [1.0, 0.8, 0.1, 0.1],
        [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ])
    params = {**_BASE_PARAMS, "semantic_scale": 0.5}

    # Act / When
    activations = _run_semantic_study_recall_cycle(Factory, params, dataset, features)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations.shape == (_LIST_LENGTH,)
    assert activations[-1] > activations[0]  # recency


# ── multiplicative semantic CMR ─────────────────────────────────────────────

def test_multiplicative_semantic_cmr_shows_recency():
    """Behavior: Multiplicative Semantic CMR shows temporal recency.

    Given:
      - Items [1, 2, 3, 4] studied in order.
      - Non-identity feature matrix with inter-item similarity.
    When:
      - A study-recall cycle is run with semantic_scale=0.5.
    Then:
      - Activations are finite with correct shape and recency.
    Why this matters:
      - Verifies the multiplicative semantic variant (which scales item
        activations by semantic similarity) preserves temporal recency.
    """
    # Arrange / Given
    from jaxcmr.models.multiplicative_semantic_cmr import make_factory

    Factory = make_factory(
        LinearMemory.init_mfc,
        LinearMemory.init_mcf,
        TemporalContext.init,
        PositionalTermination,
    )
    dataset: Any = {
        **_DATASET,
        "pres_itemids": jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32),
    }
    features = jnp.array([
        [1.0, 0.8, 0.1, 0.1],
        [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ])
    params = {**_BASE_PARAMS, "semantic_scale": 0.5}

    # Act / When
    activations = _run_semantic_study_recall_cycle(Factory, params, dataset, features)

    # Assert / Then
    assert jnp.all(jnp.isfinite(activations)).item()
    assert activations.shape == (_LIST_LENGTH,)
    assert activations[-1] > activations[0]  # recency
