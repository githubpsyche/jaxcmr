"""Tests for selective interference pipeline composability."""

import pytest
import jax.numpy as jnp
from jax import lax, random, vmap

from jaxcmr.selective_interference.cmr import PhasedCMR
from jaxcmr.selective_interference.paradigm import Paradigm
from jaxcmr.selective_interference.pipeline import (
    configure_rates,
    film_recalled_stats,
    prepare_sweep,
    run_count_sweep,
    run_sweep,
    sweep_rngs,
)


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

_N_FILM = 3
_N_BREAK = 2
_N_INTERF = 2
_N_FILLER = 1
_LIST_LENGTH = 10
_MAX_RECALL = 10

_PARADIGM = Paradigm(
    n_film=_N_FILM,
    n_break=_N_BREAK,
    n_interference=_N_INTERF,
    n_filler=_N_FILLER,
    max_recall=_MAX_RECALL,
    experiment_count=3,
)


def _make_cached_model():
    model = _model_factory(_PARADIGM.list_length, _BASE_PARAMS, None)
    film = _PARADIGM.film_items
    brk = _PARADIGM.break_items
    model = lax.fori_loop(
        0, film.size, lambda i, m: m.experience_film(film[i]), model,
    )
    model = lax.fori_loop(
        0, brk.size, lambda i, m: m.experience_break(brk[i]), model,
    )
    model = model.start_reminders()
    return lax.fori_loop(
        0, film.size, lambda i, m: m.remind(film[i]), model,
    )


def _interf_items():
    return jnp.arange(
        _N_FILM + _N_BREAK + 1, _N_FILM + _N_BREAK + _N_INTERF + 1
    )


def _filler_items():
    return jnp.arange(
        _N_FILM + _N_BREAK + _N_INTERF + 1,
        _N_FILM + _N_BREAK + _N_INTERF + _N_FILLER + 1,
    )


def _make_batched_params(n_subjects=2):
    return {k: jnp.full(n_subjects, v) for k, v in _BASE_PARAMS.items()}


# ── Phase encoder tests ─────────────────────────────────────────────


def test_advances_study_index_when_film_encoded():
    """Behavior: experience_film advances study_index by number of items.

    Given:
      - A fresh model with study_index == 0
    When:
      - experience_film is called for each film item
    Then:
      - study_index equals n_film
    Why this matters:
      - Study index tracks encoding position for primacy and recallability
    """
    # Arrange / Given
    model = _model_factory(_PARADIGM.list_length, _BASE_PARAMS, None)
    film = _PARADIGM.film_items

    # Act / When
    result = lax.fori_loop(
        0, film.size, lambda i, m: m.experience_film(film[i]), model,
    )

    # Assert / Then
    assert int(result.study_index) == _N_FILM


def test_advances_study_index_when_break_encoded():
    """Behavior: experience_break advances study_index by number of items.

    Given:
      - A model after film encoding with study_index == n_film
    When:
      - experience_break is called for each break item
    Then:
      - study_index equals n_film + n_break
    Why this matters:
      - Break items occupy study positions after film items
    """
    # Arrange / Given
    model = _model_factory(_PARADIGM.list_length, _BASE_PARAMS, None)
    film = _PARADIGM.film_items
    brk = _PARADIGM.break_items
    model = lax.fori_loop(
        0, film.size, lambda i, m: m.experience_film(film[i]), model,
    )

    # Act / When
    result = lax.fori_loop(
        0, brk.size, lambda i, m: m.experience_break(brk[i]), model,
    )

    # Assert / Then
    assert int(result.study_index) == _N_FILM + _N_BREAK


def test_preserves_study_index_when_reminders_run():
    """Behavior: remind does not advance study_index.

    Given:
      - A model after film + break encoding
    When:
      - start_reminders + remind replays film items
    Then:
      - study_index is unchanged (reminders don't encode new items)
      - Context state changes (reminder reinstates associations)
    Why this matters:
      - Reminders update context without consuming study positions
    """
    # Arrange / Given
    model = _model_factory(_PARADIGM.list_length, _BASE_PARAMS, None)
    film = _PARADIGM.film_items
    brk = _PARADIGM.break_items
    model = lax.fori_loop(
        0, film.size, lambda i, m: m.experience_film(film[i]), model,
    )
    model = lax.fori_loop(
        0, brk.size, lambda i, m: m.experience_break(brk[i]), model,
    )
    pre_index = int(model.study_index)
    pre_context = model.context.state.copy()

    # Act / When
    model = model.start_reminders()
    result = lax.fori_loop(
        0, film.size, lambda i, m: m.remind(film[i]), model,
    )

    # Assert / Then
    assert int(result.study_index) == pre_index
    assert not jnp.allclose(result.context.state, pre_context)


def test_marks_items_recallable_when_interference_encoded():
    """Behavior: experience_interference marks items as recallable.

    Given:
      - A cached model after film + break + reminder
    When:
      - experience_interference encodes 2 interference items
    Then:
      - The interference item indices are marked recallable
    Why this matters:
      - Items must be recallable to appear in retrieval competition
    """
    # Arrange / Given
    cached = _make_cached_model()
    interf = _interf_items()

    # Act / When
    result = lax.fori_loop(
        0, interf.size,
        lambda i, m: m.experience_interference(interf[i]), cached,
    )

    # Assert / Then
    for item_id in interf:
        assert bool(result.recallable[int(item_id) - 1])


def test_marks_items_recallable_when_filler_encoded():
    """Behavior: experience_filler marks encoded items as recallable.

    Given:
      - A cached model after film + break + reminder + interference
    When:
      - experience_filler encodes 1 filler item
    Then:
      - The filler item index is marked recallable
    Why this matters:
      - Filler items participate in retrieval competition
    """
    # Arrange / Given
    cached = _make_cached_model()
    interf = _interf_items()
    cached = lax.fori_loop(
        0, interf.size,
        lambda i, m: m.experience_interference(interf[i]), cached,
    )
    filler = _filler_items()

    # Act / When
    result = lax.fori_loop(
        0, filler.size,
        lambda i, m: m.experience_filler(filler[i]), cached,
    )

    # Assert / Then
    for item_id in filler:
        assert bool(result.recallable[int(item_id) - 1])


def test_skips_zero_padded_items_when_film_encoded():
    """Behavior: zero entries in item arrays are not encoded.

    Given:
      - A film item array with trailing zeros (zero-padded)
    When:
      - experience_film processes the padded array
    Then:
      - study_index advances only for nonzero items
      - Zero-index slot remains not recallable
    Why this matters:
      - Sweep tiers use zero-padding for variable phase sizes under JIT
    """
    # Arrange / Given
    model = _model_factory(_PARADIGM.list_length, _BASE_PARAMS, None)
    padded = jnp.array([1, 2, 0])  # 2 real items + 1 zero pad

    # Act / When
    result = lax.fori_loop(
        0, padded.size, lambda i, m: m.experience_film(padded[i]), model,
    )

    # Assert / Then
    assert int(result.study_index) == 2
    assert bool(result.recallable[0])  # item 1
    assert bool(result.recallable[1])  # item 2


# ── Trial generation + batching tests ────────────────────────────────


@pytest.mark.parametrize("cache_after", [
    "creation", "film", "break", "reminder", "interference", "filler",
])
def test_returns_correct_shape_when_cache_after_varies(cache_after):
    """Behavior: sweep produces correct shape for all cache points.

    Given:
      - A prepared sweep at the given cache point with 2 subjects
    When:
      - run_sweep sweeps tau_scale over a single value
    Then:
      - Output shape is (1, n_subjects, experiment_count, max_recall)
    Why this matters:
      - Every cache point must produce the same output format
    """
    # Arrange / Given
    n_subjects = 2
    params = _make_batched_params(n_subjects)
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after=cache_after,
    )
    rng = random.PRNGKey(42)

    # Act / When
    recalls, _ = run_sweep(prepared, rng, tau_scale=[1.0])

    # Assert / Then
    assert recalls.shape == (
        1, n_subjects, _PARADIGM.experiment_count, _MAX_RECALL,
    )


def test_recalls_contain_valid_item_ids_when_sweep_runs():
    """Behavior: all nonzero recalls reference encoded items.

    Given:
      - A prepared sweep cached after reminder
    When:
      - run_sweep sweeps interference_mcf_scale over a single value
    Then:
      - Every nonzero recall is between 1 and n_encoded inclusive
    Why this matters:
      - Invalid item IDs would corrupt SPC and recall statistics
    """
    # Arrange / Given
    n_encoded = _N_FILM + _N_BREAK + _N_INTERF + _N_FILLER
    params = _make_batched_params()
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )
    rng = random.PRNGKey(0)

    # Act / When
    recalls, _ = run_sweep(prepared, rng, interference_mcf_scale=[1.0])

    # Assert / Then
    nonzero = recalls[recalls > 0]
    assert jnp.all(nonzero >= 1)
    assert jnp.all(nonzero <= n_encoded)


def test_produces_subjects_by_reps_shape_when_batched():
    """Behavior: prepare_sweep + run_sweep produce subjects x reps shape.

    Given:
      - 2 subjects, 3 replications, cached after reminder
    When:
      - run_sweep sweeps interference_mcf_scale over 1 value
    Then:
      - Output shape is (1, n_subjects, n_reps, max_recall)
    Why this matters:
      - Correct double-vmap axes are essential for sweep execution
    """
    # Arrange / Given
    n_subjects = 2
    params = _make_batched_params(n_subjects)
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )
    rng = random.PRNGKey(0)

    # Act / When
    recalls, _ = run_sweep(prepared, rng, interference_mcf_scale=[1.0])

    # Assert / Then
    assert recalls.shape == (
        1, n_subjects, _PARADIGM.experiment_count, _MAX_RECALL,
    )


def test_matches_manual_encoding_when_cached_after_reminder():
    """Behavior: prepare_sweep encoding matches manual phase composition.

    Given:
      - A paradigm with film, break, reminder phases
    When:
      - prepare_sweep caches after reminder
    Then:
      - Cached context state matches manually composed encoding
    Why this matters:
      - Validates _encode_prefix + _PHASES table produces correct model state
    """
    # Arrange / Given
    params = _make_batched_params()
    ll = _PARADIGM.list_length
    models = vmap(lambda p: _model_factory(ll, p, None))(params)
    configured = configure_rates(models)
    film = _PARADIGM.film_items
    brk = _PARADIGM.break_items

    def _encode_through_reminder(m):
        m = lax.fori_loop(0, film.size, lambda i, m: m.experience_film(film[i]), m)
        m = lax.fori_loop(0, brk.size, lambda i, m: m.experience_break(brk[i]), m)
        m = m.start_reminders()
        return lax.fori_loop(0, film.size, lambda i, m: m.remind(film[i]), m)

    manual = vmap(_encode_through_reminder)(configured)

    # Act / When
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )

    # Assert / Then
    assert jnp.allclose(prepared.models.context.state, manual.context.state)


# ── Rate configuration tests ─────────────────────────────────────────


def test_applies_interference_scales_when_configured():
    """Behavior: configure_rates scales interference rates across subjects.

    Given:
      - 2 models with encoding_drift_rate = 0.5
    When:
      - configure_rates applies interference_drift_scale = 0.5
    Then:
      - interference_drift_rate becomes 0.25 for all subjects
      - Unspecified filler_drift_rate defaults to encoding_drift_rate
    Why this matters:
      - Sweep iterations rely on correct per-condition rate scaling
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Act / When
    configured = configure_rates(models, interference_drift_scale=0.5)

    # Assert / Then
    assert jnp.allclose(configured.interference_drift_rate, 0.25)
    assert jnp.allclose(configured.filler_drift_rate, 0.5)


def test_applies_break_scales_when_configured():
    """Behavior: configure_rates scales break rates across subjects.

    Given:
      - 2 models with encoding_drift_rate = 0.5
    When:
      - configure_rates applies break_drift_scale = 0.6
    Then:
      - break_drift_rate becomes 0.3 for all subjects
    Why this matters:
      - Break drift sweeps require correct scaling from base rate
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Act / When
    configured = configure_rates(models, break_drift_scale=0.6)

    # Assert / Then
    assert jnp.allclose(configured.break_drift_rate, 0.3)


def test_applies_reminder_scales_when_configured():
    """Behavior: configure_rates scales reminder rates across subjects.

    Given:
      - 2 models with start_drift_rate = 0.4, encoding_drift_rate = 0.5
    When:
      - configure_rates applies reminder_start_drift_scale = 0.5
        and reminder_drift_scale = 0.8
    Then:
      - reminder_start_drift_rate becomes 0.2
      - reminder_drift_rate becomes 0.4
    Why this matters:
      - Reminder phase uses distinct base rates for start vs encoding
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Act / When
    configured = configure_rates(
        models,
        reminder_start_drift_scale=0.5,
        reminder_drift_scale=0.8,
    )

    # Assert / Then
    assert jnp.allclose(configured.reminder_start_drift_rate, 0.2)
    assert jnp.allclose(configured.reminder_drift_rate, 0.4)


def test_changes_recall_when_primacy_scale_swept():
    """Behavior: sweeping primacy_scale produces different recall outcomes.

    Given:
      - A prepared sweep cached after creation
    When:
      - run_sweep sweeps primacy_scale over [0.0, 10.0]
    Then:
      - Recall arrays differ between the two sweep values
    Why this matters:
      - Confirms on-the-fly mcf_learning_rate flows through the sweep pipeline
    """
    # Arrange / Given
    params = _make_batched_params()
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="creation",
    )
    rng = random.PRNGKey(0)

    # Act / When
    recalls, _ = run_sweep(prepared, rng, primacy_scale=[0.0, 10.0])

    # Assert / Then
    assert not jnp.array_equal(recalls[0], recalls[1])


def test_applies_encoding_drift_scale_when_configured():
    """Behavior: configure_rates scales encoding drift for film phase.

    Given:
      - 2 models with encoding_drift_rate = 0.5
    When:
      - configure_rates applies encoding_drift_scale = 0.6
    Then:
      - encoding_drift_rate becomes 0.3
    Why this matters:
      - Encoding drift sweeps require scaling the film-phase rate
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Act / When
    configured = configure_rates(models, encoding_drift_scale=0.6)

    # Assert / Then
    assert jnp.allclose(configured.encoding_drift_rate, 0.3)


@pytest.mark.parametrize("scale_name,attr_name,expected", [
    ("filler_drift_scale", "filler_drift_rate", 0.3),
    ("start_drift_scale", "start_drift_rate", 0.24),
    ("tau_scale", "mcf_sensitivity", 1.2),
    ("break_mcf_scale", "break_mcf_scale", 0.6),
    ("filler_mcf_scale", "filler_mcf_scale", 0.6),
    ("primacy_decay", "primacy_decay", 0.6),
])
def test_applies_remaining_scales_when_configured(scale_name, attr_name, expected):
    """Behavior: configure_rates applies each scale correctly.

    Given:
      - 2 models with encoding_drift_rate=0.5, start_drift_rate=0.4,
        choice_sensitivity=2.0, primacy_decay=0.1
    When:
      - configure_rates applies the given scale at 0.6
    Then:
      - The corresponding model attribute matches the expected value
    Why this matters:
      - Every configurable rate must scale correctly for sweep accuracy
    """
    # Arrange / Given
    params = _make_batched_params()
    models = vmap(lambda p: _model_factory(_LIST_LENGTH, p, None))(params)

    # Act / When
    configured = configure_rates(models, **{scale_name: 0.6})

    # Assert / Then
    assert jnp.allclose(getattr(configured, attr_name), expected)


# ── Sweep helper tests ───────────────────────────────────────────────


def test_returns_correct_shape_when_sweep_rngs():
    """Behavior: sweep_rngs produces (n_subjects, n_reps) key array.

    Given:
      - 3 subjects and 5 replications
    When:
      - sweep_rngs splits the parent key
    Then:
      - Key array has shape (3, 5, ...)
      - Returned next_rng is a valid key
    Why this matters:
      - run_sweep relies on correctly shaped RNG arrays for vmap
    """
    # Arrange / Given
    rng = random.PRNGKey(0)

    # Act / When
    rngs_2d, next_rng = sweep_rngs(rng, 3, 5)

    # Assert / Then
    assert rngs_2d.shape[:2] == (3, 5)
    assert next_rng.shape == rng.shape


def test_returns_mean_and_ci_when_film_recalled_stats():
    """Behavior: film_recalled_stats computes mean film items recalled.

    Given:
      - 6 trials (2 subjects x 3 reps) all recalling exactly 3 film items
    When:
      - film_recalled_stats is computed
    Then:
      - Mean equals 3.0
      - CI bounds satisfy lower <= mean <= upper
    Why this matters:
      - Downstream plotting and analysis depend on correct summary stats
    """
    # Arrange / Given
    recalls = jnp.broadcast_to(
        jnp.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0]),
        (6, 10),
    )

    # Act / When
    mu, lo, hi = film_recalled_stats(recalls, _PARADIGM, 2)

    # Assert / Then
    assert mu == pytest.approx(3.0)
    assert lo <= mu <= hi


# ── High-level sweep API tests ────────────────────────────────────────


def test_returns_stacked_recalls_when_sweep_runs():
    """Behavior: run_sweep returns stacked recall array across sweep values.

    Given:
      - A prepared sweep cached after reminder with 2 subjects
    When:
      - run_sweep sweeps interference_mcf_scale over 2 values
    Then:
      - Output shape is (2, n_subjects, experiment_count, max_recall)
    Why this matters:
      - Correct shape is essential for downstream analysis
    """
    # Arrange / Given
    n_subjects = 2
    params = _make_batched_params(n_subjects)
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )
    rng = random.PRNGKey(0)

    # Act / When
    recalls, _ = run_sweep(
        prepared, rng, interference_mcf_scale=[0.5, 2.0],
    )

    # Assert / Then
    assert recalls.shape == (
        2, n_subjects, _PARADIGM.experiment_count, _MAX_RECALL,
    )


def test_reuses_cache_when_sweep_runs_twice():
    """Behavior: same PreparedSweep can be reused for different sweeps.

    Given:
      - A prepared sweep cached after reminder
    When:
      - run_sweep is called twice with different swept parameters
    Then:
      - Both return correct shapes without error
    Why this matters:
      - Cache reuse avoids redundant model creation and encoding
    """
    # Arrange / Given
    params = _make_batched_params()
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )
    rng = random.PRNGKey(0)

    # Act / When
    recalls1, rng = run_sweep(
        prepared, rng, interference_mcf_scale=[0.5, 1.0],
    )
    recalls2, _ = run_sweep(
        prepared, rng, interference_drift_scale=[0.5, 1.0],
    )

    # Assert / Then
    assert recalls1.shape == recalls2.shape


# ── Tier geometry tests ──────────────────────────────────────────────

_TIER_PARADIGM = Paradigm(
    n_film=3, n_break=2, n_interference=2, n_filler=1,
    n_break_max=4, n_interference_max=4, n_filler_max=3,
    max_recall=10, experiment_count=3,
)


@pytest.mark.parametrize("tier,ll_prop,item_props", [
    ("standard", "list_length",
     ["film_items", "break_items", "interference_items", "filler_items"]),
    ("break_extended", "break_extended_list_length",
     ["film_items", "break_extended_break_items",
      "break_extended_interference_items", "break_extended_filler_items"]),
    ("interference_extended", "interference_extended_list_length",
     ["film_items", "break_items",
      "interference_extended_interference_items",
      "interference_extended_filler_items"]),
    ("filler_extended", "filler_extended_list_length",
     ["film_items", "break_items", "interference_items",
      "filler_extended_filler_items"]),
])
def test_phase_items_are_disjoint_and_within_bounds_when_tier_selected(
    tier, ll_prop, item_props,
):
    """Behavior: each tier's phase items are disjoint and within list_length.

    Given:
      - A paradigm with distinct standard and max phase sizes
    When:
      - Item arrays and list_length are resolved for the given tier
    Then:
      - No item ID appears in more than one phase
      - Every item ID is between 1 and list_length inclusive
    Why this matters:
      - Overlapping IDs cause phase collisions; out-of-range IDs crash encoding
    """
    # Arrange / Given
    paradigm = _TIER_PARADIGM
    ll = getattr(paradigm, ll_prop)

    # Act / When
    all_ids = []
    for prop in item_props:
        ids = [int(x) for x in getattr(paradigm, prop) if x > 0]
        all_ids.extend(ids)

    # Assert / Then
    assert len(all_ids) == len(set(all_ids)), (
        f"Tier {tier!r}: duplicate IDs across phases"
    )
    assert all(1 <= i <= ll for i in all_ids), (
        f"Tier {tier!r}: IDs outside [1, {ll}]"
    )


@pytest.mark.parametrize("tier,ll_prop", [
    ("break_extended", "break_extended_list_length"),
    ("interference_extended", "interference_extended_list_length"),
    ("filler_extended", "filler_extended_list_length"),
])
def test_uses_extended_list_length_when_tier_specified(tier, ll_prop):
    """Behavior: prepare_sweep creates models sized for the tier's list_length.

    Given:
      - A paradigm where extended tiers have larger list_length than standard
    When:
      - prepare_sweep is called with the given tier
    Then:
      - The model's recallable array matches the tier's list_length
    Why this matters:
      - Using standard list_length for extended tiers causes out-of-bounds indexing
    """
    # Arrange / Given
    paradigm = _TIER_PARADIGM
    expected_ll = getattr(paradigm, ll_prop)
    params = _make_batched_params()

    # Act / When
    prepared = prepare_sweep(
        params, paradigm, _model_factory, cache_after="creation", tier=tier,
    )

    # Assert / Then
    assert prepared.models.recallable.shape[-1] == expected_ll


@pytest.mark.parametrize("tier,extended_prop", [
    ("break_extended", "break_extended_break_items"),
    ("interference_extended", "interference_extended_interference_items"),
    ("filler_extended", "filler_extended_filler_items"),
])
def test_uses_extended_item_arrays_when_tier_specified(tier, extended_prop):
    """Behavior: prepare_sweep uses the tier's item arrays in item_args.

    Given:
      - A paradigm where extended tiers override standard item arrays
    When:
      - prepare_sweep is called with the given tier, cached after creation
    Then:
      - item_args contains the extended item array
    Why this matters:
      - Using standard items for extended tiers causes ID collisions across phases
    """
    # Arrange / Given
    paradigm = _TIER_PARADIGM
    expected_items = getattr(paradigm, extended_prop)
    params = _make_batched_params()

    # Act / When
    prepared = prepare_sweep(
        params, paradigm, _model_factory, cache_after="creation", tier=tier,
    )

    # Assert / Then
    found = any(
        isinstance(arg, jnp.ndarray) and arg.ndim > 0
        and jnp.array_equal(arg, expected_items)
        for arg in prepared.item_args
    )
    assert found, (
        f"Expected {extended_prop} in item_args for tier {tier!r}"
    )


def test_raises_when_invalid_tier_specified():
    """Behavior: prepare_sweep rejects unknown tier names.

    Given:
      - A valid paradigm and parameters
    When:
      - prepare_sweep is called with tier="nonexistent"
    Then:
      - ValueError is raised
    Why this matters:
      - Typos in tier names should fail fast rather than silently using standard
    """
    # Arrange / Given
    params = _make_batched_params()

    # Act / When / Then
    with pytest.raises(ValueError, match="Unknown tier"):
        prepare_sweep(
            params, _TIER_PARADIGM, _model_factory,
            cache_after="creation", tier="nonexistent",
        )


# ── item_slots tests ────────────────────────────────────────────────


@pytest.mark.parametrize("cache_after,expected_attrs", [
    ("creation", [
        "film_items", "break_items", "interference_items", "filler_items",
    ]),
    ("film", ["break_items", "film_items", "interference_items", "filler_items"]),
    ("reminder", ["interference_items", "filler_items"]),
    ("filler", []),
])
def test_item_slots_populated_when_sweep_prepared(cache_after, expected_attrs):
    """Behavior: item_slots maps post-cache attrs to sequential indices.

    Given:
      - A paradigm and cache_after point
    When:
      - prepare_sweep is called
    Then:
      - item_slots contains exactly the deduplicated post-cache attrs
      - Indices are sequential starting from 0
    Why this matters:
      - run_count_sweep relies on item_slots to find the correct override slot
    """
    # Arrange / Given
    params = _make_batched_params()

    # Act / When
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after=cache_after,
    )

    # Assert / Then
    # film_items appears in both film and reminder phases; deduplicate
    unique = list(dict.fromkeys(expected_attrs))
    assert set(prepared.item_slots.keys()) == set(unique)
    for attr in unique:
        assert prepared.item_slots[attr] == unique.index(attr)


# ── Item-count sweep tests ──────────────────────────────────────────


def test_stacks_recalls_when_count_sweep_run():
    """Behavior: run_count_sweep returns stacked recalls across count values.

    Given:
      - A prepared sweep with interference_extended tier
    When:
      - run_count_sweep sweeps n_interference over 2 values
    Then:
      - Output shape is (2, n_subjects, experiment_count, max_recall)
    Why this matters:
      - Correct shape is essential for downstream remap and SPC
    """
    # Arrange / Given
    n_subjects = 2
    params = _make_batched_params(n_subjects)
    paradigm = _TIER_PARADIGM
    prepared = prepare_sweep(
        params, paradigm, _model_factory,
        cache_after="reminder", tier="interference_extended",
    )
    rng = random.PRNGKey(42)

    # Act / When
    recalls, _ = run_count_sweep(
        prepared, rng, paradigm, "n_interference", [2, 4],
    )

    # Assert / Then
    assert recalls.shape == (
        2, n_subjects, paradigm.experiment_count, paradigm.max_recall,
    )


def test_overrides_correct_slot_when_filler_count_swept():
    """Behavior: run_count_sweep overrides the filler slot, not interference.

    Given:
      - A prepared sweep cached after reminder (item_args: interf, filler, max)
    When:
      - run_count_sweep sweeps n_filler over 2 values
    Then:
      - Output has correct shape (confirming slot 1 was overridden)
      - Results differ between count values
    Why this matters:
      - Filler items occupy slot 1, not slot 0; wrong slot corrupts simulation
    """
    # Arrange / Given
    n_subjects = 2
    params = _make_batched_params(n_subjects)
    paradigm = _TIER_PARADIGM
    prepared = prepare_sweep(
        params, paradigm, _model_factory,
        cache_after="reminder", tier="filler_extended",
    )
    rng = random.PRNGKey(42)

    # Act / When
    recalls, _ = run_count_sweep(
        prepared, rng, paradigm, "n_filler", [1, 3],
    )

    # Assert / Then
    assert recalls.shape == (
        2, n_subjects, paradigm.experiment_count, paradigm.max_recall,
    )
    assert not jnp.array_equal(recalls[0], recalls[1])


def test_raises_when_invalid_phase_specified():
    """Behavior: run_count_sweep rejects unknown phase names.

    Given:
      - A valid prepared sweep
    When:
      - run_count_sweep is called with phase="n_reminder"
    Then:
      - ValueError is raised
    Why this matters:
      - Only break, interference, filler phases support count sweeps
    """
    # Arrange / Given
    params = _make_batched_params()
    prepared = prepare_sweep(
        params, _PARADIGM, _model_factory, cache_after="reminder",
    )
    rng = random.PRNGKey(0)

    # Act / When / Then
    with pytest.raises(ValueError, match="Unknown phase"):
        run_count_sweep(prepared, rng, _PARADIGM, "n_reminder", [1, 2])
