"""Tests for selective interference model creation and composition."""

import jax.numpy as jnp
from jax import lax, vmap

from jaxcmr.selective_interference.cmr import PhasedCMR
from jaxcmr.selective_interference.paradigm import Paradigm, make_is_emotional
from jaxcmr.selective_interference.pipeline import configure_rates


def _model_factory(list_length, parameters, _connections):
    return PhasedCMR(list_length, parameters)


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
}

_N_SUBJECTS = 2
_N_FILM = 3
_N_BREAK = 2
_LIST_LENGTH = 10


def _make_batched_params():
    return {k: jnp.full(_N_SUBJECTS, v) for k, v in _BASE_PARAMS.items()}


def test_creates_batched_models_when_called():
    """Behavior: vmap model construction produces one model per subject.

    Given:
      - Per-subject parameter arrays for 2 subjects
    When:
      - Model factory is vmapped over params
    Then:
      - Returned model has batched leading dimension equal to n_subjects
      - All study indices are 0 (no items encoded yet)
      - Context state is finite
    Why this matters:
      - Verifies vmap over params dict works for model construction
    """
    # Arrange / Given
    params = _make_batched_params()

    # Act / When
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Assert / Then
    assert models.study_index.shape == (_N_SUBJECTS,)
    assert jnp.all(models.study_index == 0)
    assert jnp.all(jnp.isfinite(models.context.state))


def test_context_differs_from_initial_when_phases_composed():
    """Behavior: composing phases changes context away from initial state.

    Given:
      - Batched models with default rates configured
    When:
      - Film, break, and reminder phases are composed via vmap
    Then:
      - Context state differs from the initial context state
    Why this matters:
      - Confirms encoding and reminder actually modify context
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)
    configured = configure_rates(models)
    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    # Act / When
    def _encode_through_reminder(m):
        m = lax.fori_loop(
            0, film_items.size,
            lambda i, m: m.experience_film(film_items[i]), m,
        )
        m = lax.fori_loop(
            0, break_items.size,
            lambda i, m: m.experience_break(break_items[i]), m,
        )
        m = m.start_reminders()
        return lax.fori_loop(
            0, film_items.size, lambda i, m: m.remind(film_items[i]), m,
        )

    cached = vmap(_encode_through_reminder)(configured)

    # Assert / Then
    assert not jnp.allclose(cached.context.state, cached.context.initial_state)


def test_film_and_break_items_are_recallable_when_composed():
    """Behavior: all film and break items are marked recallable after encoding.

    Given:
      - 3 film items and 2 break items
    When:
      - Film and break phases are composed via vmap
    Then:
      - Recallable flags are True for film and break item indices
      - Recallable flags are False for unencoded item indices
    Why this matters:
      - Confirms experience() was called for every film and break item
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)
    configured = configure_rates(models)
    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    # Act / When
    def _encode_film_and_break(m):
        m = lax.fori_loop(
            0, film_items.size,
            lambda i, m: m.experience_film(film_items[i]), m,
        )
        return lax.fori_loop(
            0, break_items.size,
            lambda i, m: m.experience_break(break_items[i]), m,
        )

    cached = vmap(_encode_film_and_break)(configured)

    # Assert / Then
    encoded_count = _N_FILM + _N_BREAK
    for s in range(_N_SUBJECTS):
        assert jnp.all(cached.recallable[s, :encoded_count])
        assert not jnp.any(cached.recallable[s, encoded_count:])


# ── Emotional mechanism tests ─────────────────────────────────────


_EMO_PARAMS = {**_BASE_PARAMS, "emotion_scale": 2.0}


def test_emotional_mcf_nonzero_when_emotional_items_encoded():
    """Behavior: emotional_mcf holds nonzero values for emotional items after encoding.

    Given:
      - A model with emotion_scale=2.0 and first 3 items emotional
    When:
      - All items are encoded via experience_film
    Then:
      - emotional_mcf is nonzero at emotional item positions
      - emotional_mcf is zero at neutral item positions
    Why this matters:
      - Verifies the encoding path of the emotional mechanism
    """
    # Arrange / Given
    is_emotional = jnp.zeros(_LIST_LENGTH).at[:_N_FILM].set(1.0)
    model = PhasedCMR(_LIST_LENGTH, _EMO_PARAMS, is_emotional=is_emotional)
    film_items = jnp.arange(1, _N_FILM + 1)

    # Act / When
    encoded = lax.fori_loop(
        0, film_items.size,
        lambda i, m: m.experience_film(film_items[i]), model,
    )

    # Assert / Then
    assert jnp.all(encoded.emotional_mcf[:_N_FILM] > 0)
    assert jnp.all(encoded.emotional_mcf[_N_FILM:] == 0)


def test_emotional_mcf_zero_when_neutral_items_encoded():
    """Behavior: emotional_mcf stays zero for neutral items despite positive emotion_scale.

    Given:
      - A model with emotion_scale=2.0 and all items neutral
    When:
      - Items are encoded via experience_film
    Then:
      - emotional_mcf is zero everywhere
    Why this matters:
      - Verifies that emotion_scale alone does not create spurious emotional support
    """
    # Arrange / Given
    is_emotional = jnp.zeros(_LIST_LENGTH)
    model = PhasedCMR(_LIST_LENGTH, _EMO_PARAMS, is_emotional=is_emotional)
    film_items = jnp.arange(1, _N_FILM + 1)

    # Act / When
    encoded = lax.fori_loop(
        0, film_items.size,
        lambda i, m: m.experience_film(film_items[i]), model,
    )

    # Assert / Then
    assert jnp.all(encoded.emotional_mcf == 0)


def test_activations_differ_when_last_recall_emotional():
    """Behavior: activations change when the last recalled item was emotional.

    Given:
      - Two identical models differing only in emotional mechanism,
        both with film items encoded and one emotional item recalled
    When:
      - activations() is called on each
    Then:
      - The activation vectors differ (emotional support is nonzero)
    Why this matters:
      - Verifies the retrieval-side emotional clustering pathway is active
    """
    # Arrange / Given
    is_emotional = jnp.zeros(_LIST_LENGTH).at[:_N_FILM].set(1.0)
    emo_model = PhasedCMR(_LIST_LENGTH, _EMO_PARAMS, is_emotional=is_emotional)
    neutral_model = PhasedCMR(_LIST_LENGTH, _BASE_PARAMS)

    film_items = jnp.arange(1, _N_FILM + 1)

    def _encode_and_recall_first(m):
        m = lax.fori_loop(
            0, film_items.size,
            lambda i, m: m.experience_film(film_items[i]), m,
        )
        m = m.start_retrieving()
        m = m.retrieve(film_items[0])  # recall item 1 (emotional)
        return m

    emo_ready = _encode_and_recall_first(emo_model)
    neutral_ready = _encode_and_recall_first(neutral_model)

    # Act / When
    emo_acts = emo_ready.activations()
    neutral_acts = neutral_ready.activations()

    # Assert / Then
    assert not jnp.allclose(emo_acts, neutral_acts, atol=1e-5)


def test_activations_not_boosted_when_last_recall_neutral():
    """Behavior: no emotional boost when the last recalled item was neutral.

    Given:
      - A model with emotional film items and neutral break items encoded,
        transitioned to retrieval, and one neutral item recalled
    When:
      - activations() is called
    Then:
      - Activations match the no-emotional baseline
    Why this matters:
      - Verifies that emotional support is gated by the last-recalled item
    """
    # Arrange / Given
    is_emotional = jnp.zeros(_LIST_LENGTH).at[:_N_FILM].set(1.0)
    emo_model = PhasedCMR(_LIST_LENGTH, _EMO_PARAMS, is_emotional=is_emotional)
    neutral_model = PhasedCMR(_LIST_LENGTH, _BASE_PARAMS)

    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    def _encode_and_recall_neutral(m):
        m = lax.fori_loop(
            0, film_items.size,
            lambda i, m: m.experience_film(film_items[i]), m,
        )
        m = lax.fori_loop(
            0, break_items.size,
            lambda i, m: m.experience_break(break_items[i]), m,
        )
        m = m.start_retrieving()
        m = m.retrieve(break_items[0])  # recall a neutral break item
        return m

    emo_ready = _encode_and_recall_neutral(emo_model)
    neutral_ready = _encode_and_recall_neutral(neutral_model)

    # Act / When
    emo_acts = emo_ready.activations()
    neutral_acts = neutral_ready.activations()

    # Assert / Then
    assert jnp.allclose(emo_acts, neutral_acts, atol=1e-5)


def test_configure_rates_replaces_emotion_scale_when_provided():
    """Behavior: configure_rates sets emotion_scale to the provided value.

    Given:
      - Batched models with emotion_scale=2.0
    When:
      - configure_rates is called with emotion_scale=5.0
    Then:
      - Models have emotion_scale=5.0
    Why this matters:
      - Verifies the pipeline can sweep emotion_scale for calibration
    """
    # Arrange / Given
    params = {k: jnp.full(_N_SUBJECTS, v) for k, v in _EMO_PARAMS.items()}
    is_emotional = jnp.zeros(_LIST_LENGTH).at[:_N_FILM].set(1.0)
    models = vmap(
        lambda p: PhasedCMR(_LIST_LENGTH, p, is_emotional=is_emotional)
    )(params)

    # Act / When
    configured = configure_rates(models, emotion_scale=5.0)

    # Assert / Then
    assert jnp.allclose(configured.emotion_scale, 5.0)


def test_make_is_emotional_flags_correct_positions_when_film_emotional():
    """Behavior: make_is_emotional sets film positions to 1.0 when film_emotional=True.

    Given:
      - A paradigm with n_film=4, n_break=4, n_interference=4, n_filler=4
    When:
      - make_is_emotional is called with film_emotional=True
    Then:
      - Positions 0..3 are 1.0 (film)
      - All other positions are 0.0
    Why this matters:
      - Verifies correct item-to-position mapping for the emotional flag array
    """
    # Arrange / Given
    paradigm = Paradigm(n_film=4, n_break=4, n_interference=4, n_filler=4)

    # Act / When
    flags = make_is_emotional(paradigm, film_emotional=True)

    # Assert / Then
    assert jnp.all(flags[:4] == 1.0)
    assert jnp.all(flags[4:] == 0.0)


def test_make_is_emotional_flags_correct_positions_when_interference_emotional():
    """Behavior: make_is_emotional sets interference positions to 1.0.

    Given:
      - A paradigm with n_film=4, n_break=4, n_interference=4, n_filler=4
    When:
      - make_is_emotional is called with interference_emotional=True
    Then:
      - Positions 8..11 are 1.0 (interference)
      - All other positions are 0.0
    Why this matters:
      - Verifies that interference items map to the correct array region
    """
    # Arrange / Given
    paradigm = Paradigm(n_film=4, n_break=4, n_interference=4, n_filler=4)

    # Act / When
    flags = make_is_emotional(paradigm, interference_emotional=True)

    # Assert / Then
    assert jnp.all(flags[:8] == 0.0)
    assert jnp.all(flags[8:12] == 1.0)
    assert jnp.all(flags[12:] == 0.0)
