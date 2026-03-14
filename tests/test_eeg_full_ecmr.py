"""Tests for the full eCMR model with dual context and LPP modulation."""

import jax.numpy as jnp
import pytest
from jax import lax

from jaxcmr.models_eeg.eeg_full_ecmr import eCMR


_LIST_LENGTH = 6

_BASE_PARAMS = {
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
    "modulate_emotion_by_primacy": True,
    "emotion_scale": 1.5,
    "emotion_drift_rate": 0.3,
    "lpp_main_scale": 0.0,
    "lpp_main_threshold": 0.0,
    "lpp_inter_scale": 0.0,
    "lpp_inter_threshold": 0.0,
}


def _make_model(
    is_emotional=None,
    lpp_centered=None,
    params=None,
):
    if is_emotional is None:
        is_emotional = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    if lpp_centered is None:
        lpp_centered = jnp.zeros(_LIST_LENGTH)
    if params is None:
        params = _BASE_PARAMS
    return eCMR(_LIST_LENGTH, params, is_emotional, lpp_centered)


def _encode_all(model):
    """Encode all items sequentially (1-indexed)."""
    def body(i, m):
        return m.experience(i + 1)
    return lax.fori_loop(0, model.item_count, body, model)


# ── Emotional context structure ────────────────────────────────


def test_emotional_context_has_three_units_when_initialized():
    """Behavior: emotional context has start-of-list + 2 category poles.

    Given:
      - A freshly constructed eCMR model
    When:
      - Emotional context state is inspected
    Then:
      - Emotional context has 3 units
      - Initial state is [1, 0, 0] (start-of-list)
    Why this matters:
      - TLD19 requires 2-unit localist source features plus start-of-list
    """
    # Arrange / Given
    model = _make_model()

    # Act / When
    state = model.emotion_context.state

    # Assert / Then
    assert state.shape == (3,)
    assert jnp.allclose(state, jnp.array([1.0, 0.0, 0.0]))


def test_neutral_items_update_emotional_context_when_encoded():
    """Behavior: neutral items drift emotional context toward neutral pole.

    Given:
      - A model with all-neutral items
    When:
      - All items are encoded
    Then:
      - Emotional context has drifted away from initial [1, 0, 0]
      - Neutral pole (unit 2) has positive activation
      - Emotional pole (unit 1) remains near zero
    Why this matters:
      - TLD19 requires BOTH item types to update emotional context
    """
    # Arrange / Given
    all_neutral = jnp.zeros(_LIST_LENGTH)
    model = _make_model(is_emotional=all_neutral)

    # Act / When
    model = _encode_all(model)

    # Assert / Then
    state = model.emotion_context.state
    assert state[2] > 0.1, "Neutral pole should have positive activation"
    assert state[2] > state[1], "Neutral pole should dominate emotional pole"


def test_emotional_items_update_emotional_context_when_encoded():
    """Behavior: emotional items drift emotional context toward emotional pole.

    Given:
      - A model with all-emotional items
    When:
      - All items are encoded
    Then:
      - Emotional pole (unit 1) has positive activation
      - Neutral pole (unit 2) remains near zero
    Why this matters:
      - TLD19 requires emotional items to push context toward emotional pole
    """
    # Arrange / Given
    all_emotional = jnp.ones(_LIST_LENGTH)
    model = _make_model(is_emotional=all_emotional)

    # Act / When
    model = _encode_all(model)

    # Assert / Then
    state = model.emotion_context.state
    assert state[1] > 0.1, "Emotional pole should have positive activation"
    assert state[1] > state[2], "Emotional pole should dominate neutral pole"


# ── Pre-experimental associations ──────────────────────────────


def test_emotion_mfc_has_both_categories_when_initialized():
    """Behavior: pre-experimental emotion_mfc links both categories.

    Given:
      - A model with alternating emotional/neutral items
    When:
      - emotion_mfc state is inspected
    Then:
      - Emotional items have nonzero link to emotional pole (column 1)
      - Neutral items have nonzero link to neutral pole (column 2)
      - Both have zero link to the other category's pole
    Why this matters:
      - TLD19 requires 2-unit localist: both categories have source features
    """
    # Arrange / Given
    model = _make_model()

    # Act / When
    mfc_state = model.emotion_mfc.state  # (list_length, 3)

    # Assert / Then
    lr = _BASE_PARAMS["learning_rate"]
    expected_weight = 1 - lr
    # Emotional items (0, 2, 4) link to column 1 (emotional pole)
    assert jnp.allclose(mfc_state[0, 1], expected_weight)
    assert jnp.allclose(mfc_state[0, 2], 0.0)
    # Neutral items (1, 3, 5) link to column 2 (neutral pole)
    assert jnp.allclose(mfc_state[1, 2], expected_weight)
    assert jnp.allclose(mfc_state[1, 1], 0.0)


def test_emotion_mcf_is_zero_when_initialized():
    """Behavior: pre-experimental emotion_mcf is all zeros.

    Given:
      - A freshly constructed eCMR model
    When:
      - emotion_mcf state is inspected
    Then:
      - All entries are zero
    Why this matters:
      - TLD19 MATLAB: eye_cf = 0, no pre-experimental source→item associations
    """
    # Arrange / Given
    model = _make_model()

    # Act / When
    mcf_state = model.emotion_mcf.state  # (3, list_length)

    # Assert / Then
    assert jnp.allclose(mcf_state, 0.0)


# ── Encoding pathway separation ───────────────────────────────


def test_phi_emot_scales_emotion_mcf_only_when_encoded():
    """Behavior: phi_emot modulates emotion_mcf learning, not temporal mcf.

    Given:
      - Two models: one with emotion_scale=0, one with emotion_scale=2
      - Both have the same temporal pathway parameters
    When:
      - One emotional item is encoded in each model
    Then:
      - Temporal mcf states are identical between models
      - Emotional mcf states differ between models
    Why this matters:
      - TLD19 Eq. 11: phi_emot scales L^CF_sw only, not L^CF_wt
    """
    # Arrange / Given
    params_low = {**_BASE_PARAMS, "emotion_scale": 0.0}
    params_high = {**_BASE_PARAMS, "emotion_scale": 2.0}
    is_emotional = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)

    model_low = _make_model(is_emotional=is_emotional, params=params_low)
    model_high = _make_model(is_emotional=is_emotional, params=params_high)

    # Act / When
    model_low = model_low.experience(1)   # encode emotional item 0
    model_high = model_high.experience(1)  # encode emotional item 0

    # Assert / Then
    assert jnp.allclose(
        model_low.mcf.state, model_high.mcf.state
    ), "Temporal MCF should be identical regardless of emotion_scale"
    assert not jnp.allclose(
        model_low.emotion_mcf.state, model_high.emotion_mcf.state
    ), "Emotional MCF should differ with different emotion_scale"


def test_temporal_mcf_uses_primacy_only_when_encoded():
    """Behavior: temporal MCF learning rate equals primacy at each position.

    Given:
      - A model with known primacy parameters
    When:
      - An emotional item is encoded at position 0
    Then:
      - Temporal MCF change matches primacy * outer(context, item)
      - The learning rate does NOT include phi_emot
    Why this matters:
      - TLD19 Eq. 11: temporal pathway uses primacy only
    """
    # Arrange / Given
    params = {**_BASE_PARAMS, "emotion_scale": 5.0, "learn_after_context_update": False}
    is_emotional = jnp.ones(_LIST_LENGTH)
    model = _make_model(is_emotional=is_emotional, params=params)
    mcf_before = model.mcf.state.copy()

    # Act / When
    model_after = model.experience(1)
    mcf_diff = model_after.mcf.state - mcf_before

    # Assert / Then
    # Expected learning rate: primacy at position 0 = primacy_scale * exp(0) + 1
    expected_lr = _BASE_PARAMS["primacy_scale"] * 1.0 + 1
    item = jnp.eye(_LIST_LENGTH)[0]
    context_state = model.context.state  # pre-update (learn_after=False)
    expected_diff = expected_lr * jnp.outer(context_state, item)
    assert jnp.allclose(mcf_diff, expected_diff, atol=1e-6)


# ── Primacy x emotion interaction modes ────────────────────────


def test_multiplicative_primacy_emotion_when_flag_true():
    """Behavior: primacy and phi_emot multiply when flag is True.

    Given:
      - modulate_emotion_by_primacy = True
      - Known emotion_scale and primacy parameters
    When:
      - An emotional item is encoded at position 0
    Then:
      - Emotional MCF change matches primacy * phi_emot * outer(emot_ctx, item)
    Why this matters:
      - TLD19 default is multiplicative (phi_i * phi_emot)
    """
    # Arrange / Given
    params = {
        **_BASE_PARAMS,
        "emotion_scale": 2.0,
        "modulate_emotion_by_primacy": True,
        "learn_after_context_update": False,
    }
    is_emotional = jnp.ones(_LIST_LENGTH)
    model = _make_model(is_emotional=is_emotional, params=params)
    emcf_before = model.emotion_mcf.state.copy()

    # Act / When
    model_after = model.experience(1)
    emcf_diff = model_after.emotion_mcf.state - emcf_before

    # Assert / Then
    primacy_0 = _BASE_PARAMS["primacy_scale"] * 1.0 + 1
    phi_emot = 2.0  # emotion_scale * is_emotional[0] = 2.0 * 1.0
    expected_lr = primacy_0 * phi_emot
    item = jnp.eye(_LIST_LENGTH)[0]
    emot_ctx = model.emotion_context.state  # pre-update
    expected_diff = expected_lr * jnp.outer(emot_ctx, item)
    assert jnp.allclose(emcf_diff, expected_diff, atol=1e-6)


def test_additive_primacy_emotion_when_flag_false():
    """Behavior: primacy and phi_emot add when flag is False.

    Given:
      - modulate_emotion_by_primacy = False
      - Known emotion_scale and primacy parameters
    When:
      - An emotional item is encoded at position 0
    Then:
      - Emotional MCF learning rate equals primacy + phi_emot
    Why this matters:
      - Additive mode preserves primacy contribution even when phi_emot is large
    """
    # Arrange / Given
    params = {
        **_BASE_PARAMS,
        "emotion_scale": 2.0,
        "modulate_emotion_by_primacy": False,
        "learn_after_context_update": False,
    }
    is_emotional = jnp.ones(_LIST_LENGTH)
    model = _make_model(is_emotional=is_emotional, params=params)
    emcf_before = model.emotion_mcf.state.copy()

    # Act / When
    model_after = model.experience(1)
    emcf_diff = model_after.emotion_mcf.state - emcf_before

    # Assert / Then
    primacy_0 = _BASE_PARAMS["primacy_scale"] * 1.0 + 1
    phi_emot = 2.0
    expected_lr = primacy_0 + phi_emot  # additive
    item = jnp.eye(_LIST_LENGTH)[0]
    emot_ctx = model.emotion_context.state
    expected_diff = expected_lr * jnp.outer(emot_ctx, item)
    assert jnp.allclose(emcf_diff, expected_diff, atol=1e-6)


# ── LPP modulation ────────────────────────────────────────────


def test_lpp_modulates_phi_emot_when_main_scale_nonzero():
    """Behavior: LPP main effect enters phi_emot for all items.

    Given:
      - lpp_main_scale > 0 with known LPP values
      - emotion_scale = 0 to isolate LPP effect
    When:
      - Model is constructed
    Then:
      - phi_emot reflects lpp_main_scale * (lpp - threshold)
      - Both emotional and neutral items have nonzero phi_emot
    Why this matters:
      - LPP main effect modulates encoding strength for ALL items
    """
    # Arrange / Given
    params = {**_BASE_PARAMS, "emotion_scale": 0.0, "lpp_main_scale": 1.0}
    lpp = jnp.array([0.5, -0.3, 0.2, 0.0, -0.1, 0.4])

    # Act / When
    model = _make_model(lpp_centered=lpp, params=params)

    # Assert / Then
    expected = lpp  # scale=1, threshold=0
    assert jnp.allclose(model.phi_emot, expected, atol=1e-6)


def test_lpp_interaction_only_affects_emotional_items_when_active():
    """Behavior: LPP interaction term is zero for neutral items.

    Given:
      - lpp_inter_scale > 0 with known LPP values
      - Alternating emotional/neutral items
    When:
      - Model is constructed
    Then:
      - Neutral items have zero interaction term
      - Emotional items have nonzero interaction term
    Why this matters:
      - Interaction term is lpp_inter_scale * (lpp - threshold) * is_emotional
    """
    # Arrange / Given
    params = {
        **_BASE_PARAMS,
        "emotion_scale": 0.0,
        "lpp_main_scale": 0.0,
        "lpp_inter_scale": 1.0,
    }
    is_emotional = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    lpp = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Act / When
    model = _make_model(is_emotional=is_emotional, lpp_centered=lpp, params=params)

    # Assert / Then
    # Neutral items (1, 3, 5) should have zero phi_emot
    assert jnp.allclose(model.phi_emot[1], 0.0)
    assert jnp.allclose(model.phi_emot[3], 0.0)
    # Emotional items (0, 2, 4) should have nonzero phi_emot
    assert model.phi_emot[0] != 0.0
    assert model.phi_emot[2] != 0.0


# ── Retrieval pathway ─────────────────────────────────────────


def test_emotional_pathway_contributes_to_activations_when_encoded():
    """Behavior: emotional pathway adds to retrieval activations.

    Given:
      - A model with emotional items fully encoded
      - emotion_scale > 0
    When:
      - Retrieval activations are computed after start_retrieving
    Then:
      - Activations are positive and finite for all recallable items
    Why this matters:
      - Both temporal and emotional pathways must contribute at retrieval
    """
    # Arrange / Given
    is_emotional = jnp.ones(_LIST_LENGTH)
    model = _make_model(is_emotional=is_emotional)
    model = _encode_all(model)

    # Act / When
    model = model.start_retrieving()
    acts = model.activations()

    # Assert / Then
    assert jnp.all(acts > 0), "All items should have positive activation"
    assert jnp.all(jnp.isfinite(acts)), "Activations should be finite"


def test_emotional_context_reinstated_when_item_retrieved():
    """Behavior: retrieving an item drifts emotional context toward its pole.

    Given:
      - A fully encoded model with known emotional/neutral items
      - Model is in retrieval mode
    When:
      - An emotional item (item 1, 1-indexed) is retrieved
    Then:
      - Emotional context state changes from pre-retrieval
      - Emotional pole activation increases
    Why this matters:
      - eCMR reinstates both temporal and emotional context on recall
    """
    # Arrange / Given
    is_emotional = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    model = _make_model(is_emotional=is_emotional)
    model = _encode_all(model)
    model = model.start_retrieving()
    ctx_before = model.emotion_context.state.copy()

    # Act / When
    model = model.retrieve(1)  # retrieve emotional item 0

    # Assert / Then
    ctx_after = model.emotion_context.state
    assert not jnp.allclose(ctx_before, ctx_after), (
        "Emotional context should change after retrieval"
    )
    assert ctx_after[1] > ctx_before[1], (
        "Emotional pole should increase after retrieving emotional item"
    )


def test_outcome_probabilities_sum_to_one_when_items_recallable():
    """Behavior: outcome probabilities sum to 1 during retrieval.

    Given:
      - A fully encoded model in retrieval mode
    When:
      - outcome_probabilities is called
    Then:
      - Probabilities sum to 1.0
      - All probabilities are non-negative
    Why this matters:
      - Probability distribution must be valid for likelihood computation
    """
    # Arrange / Given
    model = _make_model()
    model = _encode_all(model)

    # Act / When
    model = model.start_retrieving()
    probs = model.outcome_probabilities()

    # Assert / Then
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
    assert jnp.all(probs >= 0)
