from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
import jax.numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.conditional_crp import (
    crp,
    plot_crp,
    set_false_at_index,
    tabulate_target_trial,
    tabulate_trial,
)
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


def test_crp_source_field_filters_by_previous_recall():
    """Behavior: Source filtering counts transitions from matching prior items."""
    recalls = jnp.array([[2, 1, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[-1, 0, -1, 0]], dtype=jnp.int32),
    }
    lag_range = pres.shape[1] - 1

    result = crp(dataset, source_field="condition", source_values=[0, -1], size=3)

    assert result.shape == (2, 7)
    assert float(result[0, lag_range - 1]) == pytest.approx(1.0)
    assert float(result[0, lag_range + 1]) == pytest.approx(0.0)
    assert float(result[0, lag_range + 2]) == pytest.approx(0.0)
    assert float(result[1, lag_range + 2]) == pytest.approx(0.0)
    assert float(result[1, lag_range + 3]) == pytest.approx(1.0)


def test_crp_source_field_combines_with_should_tabulate():
    """Behavior: Existing ``_should_tabulate`` masks combine with source filters."""
    recalls = jnp.array([[2, 1, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    should_tabulate = jnp.zeros_like(recalls, dtype=bool)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "_should_tabulate": should_tabulate,
        "condition": jnp.array([[0, -1, -1, 0]], dtype=jnp.int32),
    }

    result = crp(dataset, source_field="condition", source_values=[0], size=3)

    assert jnp.all(jnp.isnan(result)).item()


def test_tabulate_target_trial_exact_condition_denominators():
    """Behavior: actual and available counts use target-specific denominators."""
    pres = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    trial = jnp.array([2, 1, 4, 0], dtype=jnp.int32)
    conditions = jnp.array([-1, 0, -1, 0], dtype=jnp.int32)
    should_tabulate = jnp.ones_like(trial, dtype=bool)
    target_values = jnp.array([-1, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_target_trial(
        trial, pres, conditions, should_tabulate, target_values, size=3
    )

    expected_actual = np.zeros((2, 7), dtype=int)
    expected_possible = np.zeros((2, 7), dtype=int)
    expected_actual[0, lag_range - 1] = 1
    expected_actual[1, lag_range + 3] = 1
    expected_possible[0, lag_range - 1] = 1
    expected_possible[0, lag_range + 1] = 1
    expected_possible[0, lag_range + 2] = 1
    expected_possible[1, lag_range + 2] = 1
    expected_possible[1, lag_range + 3] = 1

    np.testing.assert_array_equal(np.asarray(actual), expected_actual)
    np.testing.assert_array_equal(np.asarray(possible), expected_possible)


def test_crp_target_field_uses_condition_specific_denominators():
    """Behavior: Target conditioning computes CRP per available target value."""
    recalls = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[0, 0, 0]], dtype=jnp.int32),
    }
    lag_range = pres.shape[1] - 1

    result = crp(dataset, target_field="condition", target_values=[-1, 0], size=3)

    assert jnp.all(jnp.isnan(result[0])).item()
    assert float(result[1, lag_range + 1]) == pytest.approx(1.0)


def test_crp_source_and_target_conditioning():
    """Behavior: Source filters combine with target-specific denominators."""
    recalls = jnp.array([[2, 1, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[-1, 0, -1, 0]], dtype=jnp.int32),
    }
    lag_range = pres.shape[1] - 1

    result = crp(
        dataset,
        source_field="condition",
        source_values=[0],
        target_field="condition",
        target_values=[-1, 0],
        size=3,
    )

    assert result.shape == (2, 7)
    assert float(result[0, lag_range - 1]) == pytest.approx(1.0)
    assert float(result[0, lag_range + 1]) == pytest.approx(0.0)
    assert float(result[1, lag_range + 2]) == pytest.approx(0.0)


def test_crp_rejects_incomplete_condition_arguments():
    """Behavior: Field/value condition arguments must be paired."""
    dataset: Any = make_dataset(
        jnp.array([[1, 2, 0]], dtype=jnp.int32),
        jnp.array([[1, 2, 3]], dtype=jnp.int32),
    )

    with pytest.raises(ValueError):
        crp(dataset, source_field="condition", size=3)
    with pytest.raises(ValueError):
        crp(dataset, source_values=[0], size=3)
    with pytest.raises(ValueError):
        crp(dataset, target_field="condition", size=3)
    with pytest.raises(ValueError):
        crp(dataset, target_values=[0], size=3)


def test_crp_rejects_multiple_sources_with_target_conditioning():
    """Behavior: Source-target conditioning uses one source filter per call."""
    recalls = jnp.array([[2, 1, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[-1, 0, -1, 0]], dtype=jnp.int32),
    }

    with pytest.raises(ValueError):
        crp(
            dataset,
            source_field="condition",
            source_values=[0, -1],
            target_field="condition",
            target_values=[-1, 0],
            size=3,
        )


def test_plot_crp_supports_source_and_target_conditions():
    """Behavior: Plotting supports denominator-correct target conditions."""
    recalls = jnp.array([[2, 1, 4, 0], [2, 3, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres, subject=jnp.array([1, 2])),
        "condition": jnp.array([[-1, 0, -1, 0], [-1, 0, -1, 0]], dtype=jnp.int32),
    }
    mask = jnp.array([True, True])

    axis = plot_crp(
        dataset,
        mask,
        source_field="condition",
        source_values=[0],
        target_field="condition",
        target_values=[-1, 0],
        max_lag=3,
        color_cycle=["#d62728", "#1f77b4"],
        labels=["To Emotional", "To Neutral"],
        contrast_name="Target Item",
    )

    assert isinstance(axis, Axes)
    plt.close(axis.figure)


def test_crp_target_conditioning_jit_compatible():
    """Behavior: Target-conditioned CRP can be JIT compiled."""
    recalls = jnp.array([[2, 1, 4, 0]], dtype=jnp.int32)
    pres = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    dataset: Any = {
        **make_dataset(recalls, pres),
        "condition": jnp.array([[-1, 0, -1, 0]], dtype=jnp.int32),
    }
    target_values = jnp.array([-1, 0], dtype=jnp.int32)

    result_nojit = crp(dataset, target_field="condition", target_values=target_values, size=3)
    result_jit = jit(crp, static_argnames=("target_field", "size"))(
        dataset, target_field="condition", target_values=target_values, size=3
    )

    np.testing.assert_allclose(
        np.asarray(result_nojit), np.asarray(result_jit), equal_nan=True
    )
