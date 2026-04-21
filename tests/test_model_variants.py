"""Tests for all non-base CMR model variants.

Each test constructs a factory, creates a model, runs a study-recall
cycle, and verifies behavioral properties: finite activations and
temporal recency (last-studied item has highest activation).
"""

from typing import Any

import jax.numpy as jnp
import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jax import random
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.simulation import simulate_study_free_recall_and_forced_stop

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

_REPETITION_LIST = [1, 2, 1, 3]


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
    return {
        **_BASE_PARAMS,
        "mfc_sensitivity": 1.0,
        "blend_weight": blend_weight,
    }


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


def test_blend_positional_endpoint_matches_positional_before_first_recall():
    """Behavior: blend_weight=0 recovers PositionalCMR before first recall."""
    from jaxcmr.models.blend_positional_cmr import CMR as BlendCMR
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    params = _endpoint_parameters(0.0)
    positional = _run_sequence(PositionalCMR(_LIST_LENGTH, params), _REPETITION_LIST, [])
    blend = _run_sequence(BlendCMR(_LIST_LENGTH, params), _REPETITION_LIST, [])

    assert jnp.allclose(blend.outcome_probabilities(), positional.outcome_probabilities())
    assert jnp.allclose(blend.context.state, positional.context.state)


def test_blend_positional_endpoint_matches_positional_after_recall():
    """Behavior: blend_weight=0 recovers PositionalCMR after matched recall history."""
    from jaxcmr.models.blend_positional_cmr import CMR as BlendCMR
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR

    params = _endpoint_parameters(0.0)
    positional = _run_sequence(
        PositionalCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2]
    )
    blend = _run_sequence(BlendCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2])

    assert jnp.allclose(blend.outcome_probabilities(), positional.outcome_probabilities())
    assert jnp.allclose(blend.context.state, positional.context.state)


def test_blend_positional_endpoint_matches_standard_before_first_recall():
    """Behavior: blend_weight=1 recovers StandardCMR before first recall."""
    from jaxcmr.models.blend_positional_cmr import CMR as BlendCMR
    from jaxcmr.models.cmr import CMR as StandardCMR

    params = _endpoint_parameters(1.0)
    standard = _run_sequence(StandardCMR(_LIST_LENGTH, params), _REPETITION_LIST, [])
    blend = _run_sequence(BlendCMR(_LIST_LENGTH, params), _REPETITION_LIST, [])

    assert jnp.allclose(blend.outcome_probabilities(), standard.outcome_probabilities())
    assert jnp.allclose(blend.context.state, standard.context.state)


def test_blend_positional_endpoint_matches_standard_after_recall():
    """Behavior: blend_weight=1 recovers StandardCMR after matched recall history."""
    from jaxcmr.models.blend_positional_cmr import CMR as BlendCMR
    from jaxcmr.models.cmr import CMR as StandardCMR

    params = _endpoint_parameters(1.0)
    standard = _run_sequence(StandardCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2])
    blend = _run_sequence(BlendCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2])

    assert jnp.allclose(blend.outcome_probabilities(), standard.outcome_probabilities())
    assert jnp.allclose(blend.context.state, standard.context.state)


def test_blend_positional_midpoint_is_finite_and_nontrivial():
    """Behavior: interior blend weights yield valid non-endpoint behavior."""
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


def _trace_reinforcement_parameters(reinforcement: float) -> dict[str, Any]:
    """Return parameters for trace-reinforcement model tests."""
    return {
        **_BASE_PARAMS,
        "mfc_sensitivity": 1.0,
        "mcf_first_pres_reinforcement": reinforcement,
    }


def test_trace_and_probe_reinf_positional_cmr_run_simulation_path():
    """Behavior: trace-reinforcement factories support the simulation path."""
    from jaxcmr.models.probe_reinf_positional_cmr import make_factory as probe_factory
    from jaxcmr.models.trace_reinf_positional_cmr import make_factory as trace_factory

    dataset = make_dataset(
        jnp.array([[1, 0, 0, 0]], dtype=jnp.int32),
        pres_itemnos=jnp.array([[1, 2, 1, 3]], dtype=jnp.int32),
    )
    params = _trace_reinforcement_parameters(0.5)

    for make_factory in (trace_factory, probe_factory):
        Factory = make_factory(
            LinearMemory.init_mfc,
            LinearMemory.init_mcf,
            TemporalContext.init,
            PositionalTermination,
        )
        factory = Factory(dataset, None)
        model = factory.create_trial_model(jnp.int32(0), params)
        _, recalls = simulate_study_free_recall_and_forced_stop(
            model,
            dataset["pres_itemnos"][0],
            dataset["recalls"][0],
            random.PRNGKey(0),
        )

        assert recalls.shape == dataset["recalls"][0].shape
        assert jnp.all(jnp.isfinite(recalls)).item()


def test_trace_and_probe_reinf_neutral_value_matches_positional_cmr():
    """Behavior: mcf_first_pres_reinforcement=0 recovers PositionalCMR."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR
    from jaxcmr.models.probe_reinf_positional_cmr import CMR as ProbeCMR
    from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR

    params = _trace_reinforcement_parameters(0.0)
    positional = _run_sequence(
        PositionalCMR(_LIST_LENGTH, params), _REPETITION_LIST, [2]
    )

    for model_cls in (TraceCMR, ProbeCMR):
        model = _run_sequence(model_cls(_LIST_LENGTH, params), _REPETITION_LIST, [2])

        assert jnp.allclose(model.outcome_probabilities(), positional.outcome_probabilities())
        assert jnp.allclose(model.context.state, positional.context.state)
        assert jnp.allclose(model.mfc.state, positional.mfc.state)
        assert jnp.allclose(model.mcf.state, positional.mcf.state)


def test_trace_and_probe_reinf_do_not_reinforce_unique_items():
    """Behavior: non-repeated lists match PositionalCMR even with reinforcement."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR
    from jaxcmr.models.probe_reinf_positional_cmr import CMR as ProbeCMR
    from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR

    params = _trace_reinforcement_parameters(5.0)
    study_events = [1, 2, 3, 4]
    positional = _run_sequence(PositionalCMR(_LIST_LENGTH, params), study_events, [])

    for model_cls in (TraceCMR, ProbeCMR):
        model = _run_sequence(model_cls(_LIST_LENGTH, params), study_events, [])

        assert jnp.allclose(model.mcf.state, positional.mcf.state)
        assert jnp.allclose(model.outcome_probabilities(), positional.outcome_probabilities())


def test_trace_and_probe_reinf_target_first_prior_occurrence():
    """Behavior: later repetitions reinforce the first prior occurrence only."""
    from jaxcmr.models.positional_cmr import CMR as PositionalCMR
    from jaxcmr.models.probe_reinf_positional_cmr import CMR as ProbeCMR
    from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR

    params = _trace_reinforcement_parameters(2.0)
    study_events = [1, 2, 1, 3, 1]
    positional = _run_sequence(PositionalCMR(5, params), study_events, [])

    for model_cls in (TraceCMR, ProbeCMR):
        model = _run_sequence(model_cls(5, params), study_events, [])
        delta = model.mcf.state - positional.mcf.state
        other_columns = delta.at[:, 0].set(0.0)

        assert jnp.any(jnp.abs(delta[:, 0]) > 1e-6).item()
        assert jnp.allclose(other_columns, 0.0)


def test_trace_reinf_stores_pre_update_learning_state():
    """Behavior: trace variant stores the pre-update context for each position."""
    from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR

    params = _trace_reinforcement_parameters(0.5)
    model = TraceCMR(_LIST_LENGTH, params)
    pre_update_state = model.context.state

    model = model.experience(jnp.int32(1))

    assert jnp.allclose(model.trace_contexts[0], pre_update_state)


def test_trace_reinf_can_store_post_update_learning_state():
    """Behavior: trace variant can store the post-update context for each position."""
    from jaxcmr.models.trace_reinf_positional_cmr import CMR as TraceCMR

    params = _trace_reinforcement_parameters(0.5) | {
        "trace_reinforcement_after_context_update": True
    }
    model = TraceCMR(_LIST_LENGTH, params)
    pre_update_state = model.context.state

    model = model.experience(jnp.int32(1))

    assert jnp.allclose(model.trace_contexts[0], model.context.state)
    assert not jnp.allclose(model.trace_contexts[0], pre_update_state)


def test_probe_reinf_uses_current_mfc_probe_for_first_occurrence():
    """Behavior: probe variant reinforces the current MFC reconstruction."""
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
        associated_context,
        first_occurrence,
    )

    assert jnp.allclose(probe.mcf.state - positional.mcf.state, expected_delta)


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
