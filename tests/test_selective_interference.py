"""Tests for selective interference preparation pipeline."""

import jax.numpy as jnp

from jaxcmr.selective_interference.cmr import PhasedCMR
from jaxcmr.selective_interference.preparation import prepare_all_subjects


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


def test_returns_batched_models_when_jit_compiled():
    """Behavior: prepare_all_subjects produces one model per subject under JIT.

    Given:
      - Per-subject parameter arrays for 2 subjects
      - Film and break item ID arrays
    When:
      - prepare_all_subjects is called (JIT-compiled)
    Then:
      - Returned model has batched leading dimension equal to n_subjects
      - All study indices equal n_film + n_break (all items encoded)
      - Context state is finite (no NaN/Inf from encoding or reminder)
    Why this matters:
      - Verifies JIT compilation + vmap over params dict works end-to-end
    """
    # Arrange / Given
    params = _make_batched_params()
    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    # Act / When
    models = prepare_all_subjects(
        params,
        film_items,
        break_items,
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        _LIST_LENGTH,
        _model_factory,
    )

    # Assert / Then
    assert models.study_index.shape == (_N_SUBJECTS,)
    expected_study_count = _N_FILM + _N_BREAK
    assert jnp.all(models.study_index == expected_study_count)
    assert jnp.all(jnp.isfinite(models.context.state))


def test_context_differs_from_initial_when_prepared():
    """Behavior: preparation changes context away from its initial state.

    Given:
      - Per-subject parameter arrays with nonzero drift rates
    When:
      - prepare_all_subjects encodes film, break, and runs reminder
    Then:
      - Context state differs from the initial context state
    Why this matters:
      - Confirms encoding and reminder actually modify context
    """
    # Arrange / Given
    params = _make_batched_params()
    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    # Act / When
    models = prepare_all_subjects(
        params,
        film_items,
        break_items,
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        _LIST_LENGTH,
        _model_factory,
    )

    # Assert / Then
    assert not jnp.allclose(models.context.state, models.context.initial_state)


def test_film_and_break_items_are_recallable_when_prepared():
    """Behavior: all film and break items are marked recallable after preparation.

    Given:
      - 3 film items and 2 break items
    When:
      - prepare_all_subjects encodes both phases
    Then:
      - Recallable flags are True for film and break item indices
      - Recallable flags are False for unencoded item indices
    Why this matters:
      - Confirms experience() was called for every film and break item
    """
    # Arrange / Given
    params = _make_batched_params()
    film_items = jnp.arange(1, _N_FILM + 1)
    break_items = jnp.arange(_N_FILM + 1, _N_FILM + _N_BREAK + 1)

    # Act / When
    models = prepare_all_subjects(
        params,
        film_items,
        break_items,
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        jnp.float32(1.0),
        _LIST_LENGTH,
        _model_factory,
    )

    # Assert / Then
    encoded_count = _N_FILM + _N_BREAK
    for s in range(_N_SUBJECTS):
        assert jnp.all(models.recallable[s, :encoded_count])
        assert not jnp.any(models.recallable[s, encoded_count:])
