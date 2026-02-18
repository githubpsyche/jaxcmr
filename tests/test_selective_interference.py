"""Tests for selective interference model creation and composition."""

import jax.numpy as jnp
from jax import lax, vmap

from jaxcmr.selective_interference.cmr import PhasedCMR
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
