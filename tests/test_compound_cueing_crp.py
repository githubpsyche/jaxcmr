"""Tests for compound_cueing_crp analysis.

These tests verify the compound cueing analysis correctly implements the
theoretical probe for differentiating CMR (composite) from ICMR (instance-based)
memory models.

Key concepts:
    - A repeated item appears at study positions i and j (j > i).
    - Pure cueing: last two recalls are {i-2, i-1} OR {j-2, j-1}
      (both neighbors of ONE occurrence)
    - Mixed cueing: last two recalls are {j-2, i-1} OR {i-2, j-1}
      (one neighbor from each occurrence)

The analysis computes P(recall repeated item | pure cueing) vs
P(recall repeated item | mixed cueing).
"""

import jax.numpy as jnp
import pytest
from jaxcmr.analyses.compound_cueing_crp import (
    CompoundCueingTabulation,
    tabulate_trial,
    compound_cueing_crp,
)
from jaxcmr.typing import RecallDataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_dataset(
    recalls: jnp.ndarray,
    presentations: jnp.ndarray,
) -> RecallDataset:
    """Return minimal dataset keyed for the compound cueing analysis."""
    recalls = jnp.asarray(recalls, dtype=jnp.int32)
    presentations = jnp.asarray(presentations, dtype=jnp.int32)
    n_trials, _ = recalls.shape
    list_length = presentations.shape[1]
    return {
        "subject": jnp.ones((n_trials, 1), dtype=jnp.int32),
        "listLength": jnp.full((n_trials, 1), list_length, dtype=jnp.int32),
        "pres_itemnos": presentations,
        "recalls": recalls,
    }


# -----------------------------------------------------------------------------
# Basic tabulation tests
# -----------------------------------------------------------------------------


def test_no_counts_when_item_not_repeated():
    """Behavior: Non-repeated items contribute zero counts.

    Given:
        - A presentation with no repeated items.
    When:
        - tabulate_trial processes the recalls.
    Then:
        - All count values are zero.
    Why this matters:
        - The analysis should only consider repeated items.
    """
    # Arrange
    # Positions: 1=A, 2=B, 3=C, 4=D, 5=E, 6=F, 7=G, 8=H (no repeats)
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
    # Recall sequence doesn't matter since no repeats exist
    trial = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    assert counts[0] == 0, "pure_actual should be 0"
    assert counts[1] == 0, "pure_avail should be 0"
    assert counts[2] == 0, "mixed_actual should be 0"
    assert counts[3] == 0, "mixed_avail should be 0"


def test_no_counts_when_spacing_insufficient():
    """Behavior: Repeated items with insufficient spacing are ignored.

    Given:
        - An item repeated at positions 1 and 4 (spacing = 3).
        - min_spacing = 6.
    When:
        - tabulate_trial processes the recalls.
    Then:
        - All count values are zero.
    Why this matters:
        - Close repetitions don't allow for proper cueing patterns.
    """
    # Arrange
    # Item 1 at position 1 and 4 (spacing = 3 < min_spacing = 6)
    presentation = jnp.array([1, 2, 3, 1, 5, 6, 7, 8], dtype=jnp.int32)
    trial = jnp.array([2, 3, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    assert counts[0] == 0, "pure_actual should be 0"
    assert counts[1] == 0, "pure_avail should be 0"
    assert counts[2] == 0, "mixed_actual should be 0"
    assert counts[3] == 0, "mixed_avail should be 0"


def test_requires_two_prior_recalls():
    """Behavior: Compound cueing requires two prior recalls to establish context.

    Given:
        - A repeated item with sufficient spacing.
        - Only one prior recall.
    When:
        - tabulate_trial processes the recalls.
    Then:
        - No counts are registered (need two recalls to check pattern).
    Why this matters:
        - Pure and mixed cueing are defined by TWO prior recall positions.
    """
    # Arrange
    # Item 1 at positions 1 and 8 (spacing = 7 >= 6)
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 1], dtype=jnp.int32)
    # Only one prior recall (position 7) before recalling the repeated item
    trial = jnp.array([7, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    assert counts[1] == 0, "pure_avail should be 0 (need two prior recalls)"
    assert counts[3] == 0, "mixed_avail should be 0 (need two prior recalls)"


# -----------------------------------------------------------------------------
# Pure cueing tests - hand-verified scenarios
# -----------------------------------------------------------------------------


def test_pure_cueing_near_first_occurrence():
    """Behavior: Pure cueing detected when recalling neighbors of first occurrence.

    Scenario:
        - Item X appears at positions 3 and 10 (spacing = 7).
        - Recalls: positions 1, 2, then X (i.e., {i-2, i-1} = {1, 2} before X at 3).
    Expected:
        - This is pure cueing near the first occurrence.
        - If X is recalled: pure_actual = 1, pure_avail = 1.
    Why this matters:
        - Validates detection of {i-2, i-1} pattern.
    """
    # Arrange
    # Position: 1   2   3   4   5   6   7   8   9   10  11  12
    # Item:     1   2   3   4   5   6   7   8   9   3   10  11
    #                   ^-- first X=3              ^-- second X=3
    # For X=3: i=3, j=10, spacing=7
    # Pure near i: recalls at i-2=1, i-1=2, then X
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 1, then pos 2, then item 3 (the repeated item)
    trial = jnp.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert pure_avail == 1, f"Expected 1 pure opportunity, got {pure_avail}"
    assert pure_actual == 1, f"Expected 1 pure transition, got {pure_actual}"
    assert mixed_avail == 0, f"Expected 0 mixed opportunities, got {mixed_avail}"


def test_pure_cueing_near_second_occurrence():
    """Behavior: Pure cueing detected when recalling neighbors of second occurrence.

    Scenario:
        - Item X appears at positions 3 and 10 (spacing = 7).
        - Recalls: positions 8, 9, then X (i.e., {j-2, j-1} = {8, 9} before X at 10).
    Expected:
        - This is pure cueing near the second occurrence.
        - If X is recalled: pure_actual = 1, pure_avail = 1.
    Why this matters:
        - Validates detection of {j-2, j-1} pattern.
    """
    # Arrange
    # Position: 1   2   3   4   5   6   7   8   9   10  11  12
    # Item:     1   2   3   4   5   6   7   8   9   3   10  11
    # For X=3: i=3, j=10
    # Pure near j: recalls at j-2=8, j-1=9, then X
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 8, then pos 9, then item 3 (the repeated item)
    trial = jnp.array([8, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert pure_avail == 1, f"Expected 1 pure opportunity, got {pure_avail}"
    assert pure_actual == 1, f"Expected 1 pure transition, got {pure_actual}"
    assert mixed_avail == 0, f"Expected 0 mixed opportunities, got {mixed_avail}"


def test_pure_cueing_opportunity_without_transition():
    """Behavior: Pure cueing opportunity counted even without transition to repeated item.

    Scenario:
        - Item X at positions 3 and 10.
        - Recalls: positions 1, 2, then some OTHER item (not X).
    Expected:
        - pure_avail = 1 (opportunity existed)
        - pure_actual = 0 (but didn't transition to X)
    Why this matters:
        - CRP = actual / available, so we need both separately.
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 1, then pos 2, then item 4 (NOT the repeated item 3)
    trial = jnp.array([1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert pure_avail == 1, f"Expected 1 pure opportunity, got {pure_avail}"
    assert pure_actual == 0, f"Expected 0 pure transitions, got {pure_actual}"


# -----------------------------------------------------------------------------
# Mixed cueing tests - hand-verified scenarios
# -----------------------------------------------------------------------------


def test_mixed_cueing_pattern_one():
    """Behavior: Mixed cueing detected with {j-2, i-1} pattern.

    Scenario:
        - Item X at positions 3 and 10.
        - Recalls: position 8 (j-2), then position 2 (i-1), then X.
    Expected:
        - This is mixed cueing: one cue near each occurrence.
        - mixed_avail = 1, mixed_actual = 1.
    Why this matters:
        - Validates detection of {j-2, i-1} pattern.
    """
    # Arrange
    # For X=3: i=3, j=10
    # Mixed pattern 1: prev_prev = j-2 = 8, prev = i-1 = 2
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 8, then pos 2, then item 3
    trial = jnp.array([8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert mixed_avail == 1, f"Expected 1 mixed opportunity, got {mixed_avail}"
    assert mixed_actual == 1, f"Expected 1 mixed transition, got {mixed_actual}"
    assert pure_avail == 0, f"Expected 0 pure opportunities, got {pure_avail}"


def test_mixed_cueing_pattern_two():
    """Behavior: Mixed cueing detected with {i-2, j-1} pattern.

    Scenario:
        - Item X at positions 3 and 10.
        - Recalls: position 1 (i-2), then position 9 (j-1), then X.
    Expected:
        - This is mixed cueing: one cue near each occurrence.
        - mixed_avail = 1, mixed_actual = 1.
    Why this matters:
        - Validates detection of {i-2, j-1} pattern.
    """
    # Arrange
    # For X=3: i=3, j=10
    # Mixed pattern 2: prev_prev = i-2 = 1, prev = j-1 = 9
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 1, then pos 9, then item 3
    trial = jnp.array([1, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert mixed_avail == 1, f"Expected 1 mixed opportunity, got {mixed_avail}"
    assert mixed_actual == 1, f"Expected 1 mixed transition, got {mixed_actual}"
    assert pure_avail == 0, f"Expected 0 pure opportunities, got {pure_avail}"


def test_mixed_cueing_opportunity_without_transition():
    """Behavior: Mixed cueing opportunity counted even without transition.

    Scenario:
        - Item X at positions 3 and 10.
        - Recalls: position 8 (j-2), position 2 (i-1), then OTHER item (not X).
    Expected:
        - mixed_avail = 1, mixed_actual = 0.
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall item at pos 8, then pos 2, then item 4 (NOT item 3)
    trial = jnp.array([8, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert mixed_avail == 1, f"Expected 1 mixed opportunity, got {mixed_avail}"
    assert mixed_actual == 0, f"Expected 0 mixed transitions, got {mixed_actual}"


# -----------------------------------------------------------------------------
# Edge cases and availability tests
# -----------------------------------------------------------------------------


def test_repeated_item_already_recalled_not_counted():
    """Behavior: Already-recalled repeated items don't generate opportunities.

    Given:
        - A repeated item that has already been recalled.
        - Later recalls create what would be a cueing pattern.
    When:
        - tabulate_trial processes the sequence.
    Then:
        - No opportunity is counted (item unavailable).
    Why this matters:
        - Only available (unrecalled) items should generate opportunities.
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # First recall item 3, then later create a pure cueing pattern
    # But item 3 is already recalled, so no opportunity
    trial = jnp.array([3, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    # After recalling 3, the pattern {1, 2} would be pure cueing for item 3
    # But item 3 is already recalled, so no opportunity
    assert pure_avail == 0, f"Expected 0 (item already recalled), got {pure_avail}"


def test_neither_pure_nor_mixed_pattern():
    """Behavior: Random recall patterns don't count as pure or mixed.

    Scenario:
        - Item X at positions 3 and 10.
        - Recalls: positions 4, 6, then X (neither pure nor mixed pattern).
    Expected:
        - pure_avail = 0, mixed_avail = 0.
    Why this matters:
        - Only specific cueing patterns should be counted.
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)
    # Recall items at positions 4, 6 - not forming any cueing pattern
    trial = jnp.array([4, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    assert pure_avail == 0, f"Expected 0 pure opportunities, got {pure_avail}"
    assert mixed_avail == 0, f"Expected 0 mixed opportunities, got {mixed_avail}"


def test_multiple_repeated_items():
    """Behavior: Analysis handles multiple repeated items in same list.

    Scenario:
        - Item A at positions 1 and 8.
        - Item B at positions 2 and 9.
        - Recalls create pure cueing for A.
    Expected:
        - Only count the opportunity/transition for A.
    """
    # Arrange
    # Position: 1   2   3   4   5   6   7   8   9   10
    # Item:     1   2   3   4   5   6   7   1   2   8
    # Item 1: i=1, j=8 (spacing=7)
    # Item 2: i=2, j=9 (spacing=7)
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 8], dtype=jnp.int32)
    # For item 1: pure near j would be {j-2, j-1} = {6, 7}
    trial = jnp.array([6, 7, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    # Act
    counts = tabulate_trial(trial, presentation, min_spacing=6, size=2)

    # Assert
    pure_actual, pure_avail, mixed_actual, mixed_avail = counts
    # Only item 1 should match the pure pattern {6, 7}
    assert pure_avail == 1, f"Expected 1 pure opportunity for item 1, got {pure_avail}"
    assert pure_actual == 1, f"Expected 1 pure transition for item 1, got {pure_actual}"


# -----------------------------------------------------------------------------
# Full CRP computation tests
# -----------------------------------------------------------------------------


def test_crp_computation_single_trial():
    """Behavior: compound_cueing_crp correctly computes probabilities.

    Given:
        - A dataset with one trial having a pure cueing transition.
    When:
        - compound_cueing_crp is computed.
    Then:
        - pure_crp = 1.0 (1 actual / 1 available).
    """
    # Arrange
    presentation = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11]], dtype=jnp.int32)
    trial = jnp.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
    dataset = _make_dataset(trial, presentation)

    # Act
    result = compound_cueing_crp(dataset, min_spacing=6, size=2)

    # Assert
    pure_crp, mixed_crp = result
    assert float(pure_crp) == 1.0, f"Expected pure_crp=1.0, got {pure_crp}"
    assert jnp.isnan(
        mixed_crp
    ), f"Expected mixed_crp=NaN (no opportunities), got {mixed_crp}"


def test_crp_nan_when_no_opportunities():
    """Behavior: CRP is NaN when no opportunities exist for a condition.

    Given:
        - A dataset with no repeated items (no cueing opportunities).
    When:
        - compound_cueing_crp is computed.
    Then:
        - Both pure_crp and mixed_crp are NaN.
    """
    # Arrange
    presentation = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    trial = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    dataset = _make_dataset(trial, presentation)

    # Act
    result = compound_cueing_crp(dataset, min_spacing=6, size=2)

    # Assert
    pure_crp, mixed_crp = result
    assert jnp.isnan(pure_crp), f"Expected pure_crp=NaN, got {pure_crp}"
    assert jnp.isnan(mixed_crp), f"Expected mixed_crp=NaN, got {mixed_crp}"


def test_crp_aggregates_across_trials():
    """Behavior: compound_cueing_crp aggregates counts across trials.

    Given:
        - Two trials: one with pure cueing transition, one with pure cueing no transition.
    When:
        - compound_cueing_crp is computed.
    Then:
        - pure_crp = 0.5 (1 actual / 2 available).
    """
    # Arrange
    presentation = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11],
        ],
        dtype=jnp.int32,
    )
    trials = jnp.array(
        [
            [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pure: avail=1, actual=1
            [1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pure: avail=1, actual=0
        ],
        dtype=jnp.int32,
    )
    dataset = _make_dataset(trials, presentation)

    # Act
    result = compound_cueing_crp(dataset, min_spacing=6, size=2)

    # Assert
    pure_crp, mixed_crp = result
    assert abs(float(pure_crp) - 0.5) < 1e-6, f"Expected pure_crp=0.5, got {pure_crp}"


# -----------------------------------------------------------------------------
# Conceptual validation: verifying the theoretical probe
# -----------------------------------------------------------------------------


def test_pure_cueing_definition_matches_theory():
    """Conceptual: Pure cueing means both cues from ONE occurrence's neighborhood.

    The theory states:
        - For item at positions i and j
        - Pure cueing = {i-2, i-1} OR {j-2, j-1}
        - This tests whether BOTH recent cues activate the SAME memory trace.

    This test verifies the implementation matches this definition by checking
    that ONLY the exact patterns are counted, not close variations.
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)

    # Test case 1: Exact pure pattern near first occurrence {i-2, i-1} = {1, 2}
    trial_pure_i = jnp.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    counts_pure_i = tabulate_trial(trial_pure_i, presentation, min_spacing=6, size=2)

    # Test case 2: Exact pure pattern near second occurrence {j-2, j-1} = {8, 9}
    trial_pure_j = jnp.array([8, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    counts_pure_j = tabulate_trial(trial_pure_j, presentation, min_spacing=6, size=2)

    # Test case 3: Wrong order near first occurrence {i-1, i-2} = {2, 1} - NOT pure
    trial_wrong_order = jnp.array([2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    counts_wrong = tabulate_trial(
        trial_wrong_order, presentation, min_spacing=6, size=2
    )

    # Assert
    assert counts_pure_i[1] == 1, "Pattern {1,2} should be pure for item at pos 3"
    assert counts_pure_j[1] == 1, "Pattern {8,9} should be pure for item at pos 10"
    assert counts_wrong[1] == 0, "Pattern {2,1} should NOT be pure (wrong order)"


def test_mixed_cueing_definition_matches_theory():
    """Conceptual: Mixed cueing means cues from BOTH occurrences' neighborhoods.

    The theory states:
        - For item at positions i and j
        - Mixed cueing = {j-2, i-1} OR {i-2, j-1}
        - This tests whether cues from DIFFERENT traces combine.

    CMR predicts: mixed >= pure (similarities sum linearly)
    ICMR predicts: pure > mixed (sharpening before summing)
    """
    # Arrange
    presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11], dtype=jnp.int32)

    # Mixed pattern 1: {j-2, i-1} = {8, 2}
    trial_mixed_1 = jnp.array([8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    counts_mixed_1 = tabulate_trial(trial_mixed_1, presentation, min_spacing=6, size=2)

    # Mixed pattern 2: {i-2, j-1} = {1, 9}
    trial_mixed_2 = jnp.array([1, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    counts_mixed_2 = tabulate_trial(trial_mixed_2, presentation, min_spacing=6, size=2)

    # Assert
    assert counts_mixed_1[3] == 1, "Pattern {8,2} should be mixed"
    assert counts_mixed_2[3] == 1, "Pattern {1,9} should be mixed"
    # And neither should count as pure
    assert counts_mixed_1[1] == 0, "Pattern {8,2} should NOT be pure"
    assert counts_mixed_2[1] == 0, "Pattern {1,9} should NOT be pure"
