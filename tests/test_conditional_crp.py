import jax.numpy as jnp
from typing import Any

from jaxcmr.analyses.conditional_crp import set_false_at_index, tabulate_trial, crp
from jaxcmr.helpers import make_dataset


def test_clears_element_at_one_based_index():
    """Behavior: Clear the element at a 1-based index.

    Given:
      - A boolean vector ``[True, True, True]``.
      - A 1-based index of ``2``.
    When:
      - ``set_false_at_index`` is called.
    Then:
      - The second element (0-based index 1) becomes ``False``.
    Why this matters:
      - Verifies correct 1-based-to-0-based translation for availability
        tracking during recall tabulation.
    """
    # Arrange / Given
    vec = jnp.array([True, True, True], dtype=bool)
    idx = 2

    # Act / When
    updated, carry = set_false_at_index(vec, idx)

    # Assert / Then
    expected = jnp.array([True, False, True], dtype=bool)
    assert jnp.array_equal(updated, expected).item()
    assert carry is None


def test_leaves_vector_unchanged_when_index_is_zero():
    """Behavior: Index ``0`` is a no-op sentinel.

    Given:
      - A boolean vector ``[True, True, True]``.
      - An index of ``0``.
    When:
      - ``set_false_at_index`` is called.
    Then:
      - The vector is returned unchanged.
    Why this matters:
      - Zero is the padding sentinel in recall arrays; it must never
        modify the availability vector.
    """
    # Arrange / Given
    vec = jnp.array([True, True, True], dtype=bool)
    idx = 0

    # Act / When
    updated, carry = set_false_at_index(vec, idx)

    # Assert / Then
    assert jnp.array_equal(updated, vec).item()
    assert carry is None


def test_tabulate_trial_exact_lag_counts():
    """Behavior: Exact actual and available lag counts for two transitions.

    Given:
      - A trial ``[1, 3, 2, 0]`` over a 4-item canonical study list.
      - ``should_tabulate`` is all True.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - Transition 1->3 is lag +2; transition 3->2 is lag -1.
      - Actual lags have 1 at lag -1 (index 2) and 1 at lag +2 (index 5).
      - Available lags reflect items reachable at each step.
    Why this matters:
      - Verifies that the tabulator correctly accumulates both actual
        transitions and availability counts across multiple recall steps.
    """
    # Arrange / Given
    trial = jnp.array([1, 3, 2, 0], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    should_tab = jnp.ones_like(trial, dtype=bool)

    # Act / When
    actual_lags, avail_lags = tabulate_trial(trial, presentation, should_tab, size=3)

    # Assert / Then — lags: [-3, -2, -1, 0, +1, +2, +3]
    expected_actual = jnp.array([0, 0, 1, 0, 0, 1, 0])
    expected_avail = jnp.array([0, 0, 1, 0, 2, 1, 1])
    assert jnp.array_equal(actual_lags, expected_actual)
    assert jnp.array_equal(avail_lags, expected_avail)


def test_tabulate_trial_single_transition_exact_counts():
    """Behavior: Single lag +1 transition produces exact counts.

    Given:
      - A trial ``[1, 2, 0, 0]`` (one transition: 1->2, lag +1).
      - A 4-item canonical study list with all-True ``should_tabulate``.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - Actual: exactly 1 at lag +1 (index 4), zero elsewhere.
      - Available: from position 1, items 2, 3, 4 are reachable at
        lags +1, +2, +3 (indices 4, 5, 6).
    Why this matters:
      - Validates the simplest single-transition case with exact
        actual and available counts.
    """
    # Arrange / Given
    trial = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    should_tab = jnp.ones_like(trial, dtype=bool)

    # Act / When
    actual_lags, avail_lags = tabulate_trial(trial, presentation, should_tab, size=3)

    # Assert / Then — lags: [-3, -2, -1, 0, +1, +2, +3]
    expected_actual = jnp.array([0, 0, 0, 0, 1, 0, 0])
    expected_avail = jnp.array([0, 0, 0, 0, 1, 1, 1])
    assert jnp.array_equal(actual_lags, expected_actual)
    assert jnp.array_equal(avail_lags, expected_avail)


def test_crp_exact_values_for_two_trial_dataset():
    """Behavior: CRP matches hand-calculated probabilities.

    Given:
      - Trial 1: recalls ``[1, 3, 0]`` (transition lag +2).
      - Trial 2: recalls ``[2, 1, 0]`` (transition lag -1).
      - 3-item canonical presentations with all-True ``_should_tabulate``.
    When:
      - ``crp`` is called.
    Then:
      - Lag -2, 0: NaN (no available transitions at those lags).
      - Lag -1: 1.0 (trial 2 transitioned at lag -1, sole available).
      - Lag +1: 0.0 (available in both trials, never chosen).
      - Lag +2: 1.0 (trial 1 transitioned at lag +2, sole available).
    Why this matters:
      - Verifies exact CRP probabilities from cross-trial aggregation
        of actual / available lag counts.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 3, 0], [2, 1, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    should_tab = jnp.ones_like(recalls, dtype=bool)
    dataset: Any = {**make_dataset(recalls, pres), "_should_tabulate": should_tab}

    # Act / When
    result = crp(dataset, size=3)

    # Assert / Then — lags: [-2, -1, 0, +1, +2]
    assert jnp.isnan(result[0])           # lag -2: NaN
    assert jnp.isclose(result[1], 1.0)    # lag -1: 1/1
    assert jnp.isnan(result[2])           # lag  0: NaN
    assert jnp.isclose(result[3], 0.0)    # lag +1: 0/2
    assert jnp.isclose(result[4], 1.0)    # lag +2: 1/1
