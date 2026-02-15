import jax.numpy as jnp

from jaxcmr.analyses.repsrac import trial_repsrac_counts, repsrac
from jaxcmr.helpers import make_dataset


def test_trial_repsrac_counts_exact_values():
    """Behavior: Exact correct and total counts per repetition index.

    Given:
      - Presentation ``[1, 2, 1, 2]`` (items 1 and 2 each appear twice).
      - Recalls ``[1, 2, 0, 0]`` (items 1 and 2 recalled at output
        positions 1 and 2, matching their first study positions).
    When:
      - ``trial_repsrac_counts`` is called.
    Then:
      - Total: [2, 2, 0] — two items at rep index 0 (positions 1, 2),
        two at rep index 1 (positions 3, 4), none at index 2.
      - Correct: [2, 0, 0] — both items recalled at their first
        position (rep 0), but neither at their second (rep 1).
    Why this matters:
      - Verifies the exact intersection of serial position recall
        detection and repetition-index assignment.
    """
    # Arrange / Given
    pres = jnp.array([1, 2, 1, 2], dtype=jnp.int32)
    recalls = jnp.array([1, 2, 0, 0], dtype=jnp.int32)

    # Act / When
    correct, total = trial_repsrac_counts(recalls, pres, size=3)

    # Assert / Then
    assert jnp.array_equal(correct, jnp.array([2, 0, 0]))
    assert jnp.array_equal(total, jnp.array([2, 2, 0]))


def test_trial_repsrac_counts_returns_zeros_when_no_repeated_items():
    """Behavior: Return all-zero counts when no items repeat.

    Given:
      - A presentation list with unique items only.
      - Recalls that match some of those items.
    When:
      - ``trial_repsrac_counts`` is called.
    Then:
      - Both correct and total counts are all zeros because the metric
        only tracks repeated items.
    Why this matters:
      - Non-repeated items should never contribute to repetition-index
        accuracy.
    """
    # Arrange / Given
    pres = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    size = 3

    # Act / When
    correct_counts, total_counts = trial_repsrac_counts(recalls, pres, size=size)

    # Assert / Then
    assert jnp.all(correct_counts == 0), (
        f"Expected all-zero correct counts for unique items, got {correct_counts}"
    )
    assert jnp.all(total_counts == 0), (
        f"Expected all-zero total counts for unique items, got {total_counts}"
    )


def test_trial_repsrac_counts_all_correct_when_all_positions_recalled():
    """Behavior: All repeated items correct when recalled at every position.

    Given:
      - Presentation ``[1, 2, 1, 2]`` and recalls ``[1, 2, 1, 2]``
        (every item recalled in the exact study order).
    When:
      - ``trial_repsrac_counts`` is called.
    Then:
      - Correct = total = [2, 2, 0]: both rep-index 0 and 1 perfectly
        recalled.
    Why this matters:
      - Perfect serial recall of repeated items must yield 100% accuracy
        at every repetition index.
    """
    # Arrange / Given
    pres = jnp.array([1, 2, 1, 2], dtype=jnp.int32)
    recalls = jnp.array([1, 2, 1, 2], dtype=jnp.int32)

    # Act / When
    correct, total = trial_repsrac_counts(recalls, pres, size=3)

    # Assert / Then
    assert jnp.array_equal(correct, jnp.array([2, 2, 0]))
    assert jnp.array_equal(total, jnp.array([2, 2, 0]))


def test_repsrac_exact_accuracy_across_trials():
    """Behavior: Aggregate accuracy matches hand-calculated rates.

    Given:
      - Trial 1: recalls ``[1, 2, 0, 0]`` → correct at rep 0 only.
      - Trial 2: recalls ``[1, 2, 1, 2]`` → correct at both reps.
      - Presentations ``[1, 2, 1, 2]`` for both trials.
    When:
      - ``repsrac`` is called.
    Then:
      - Rep 0: (2+2)/(2+2) = 1.0 (all first-presentation items correct).
      - Rep 1: (0+2)/(2+2) = 0.5 (half of second-presentation items).
      - Rep 2: 0/0 → 0.0 (no third presentations).
    Why this matters:
      - Verifies that cross-trial aggregation of correct / total counts
        produces the correct fractional accuracy per repetition index.
    """
    # Arrange / Given
    pres = jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 0, 0], [1, 2, 1, 2]], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres)

    # Act / When
    result = repsrac(dataset, size=3)

    # Assert / Then
    expected = jnp.array([1.0, 0.5, 0.0])
    assert jnp.allclose(result, expected)
