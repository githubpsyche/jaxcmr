from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.cat_recall_by_lpp import category_lpp_recall_histogram, cat_recall_by_lpp
from jaxcmr.helpers import make_dataset


def test_category_lpp_recall_histogram_exact_bin_rates():
    """Behavior: Recall rate per LPP bin matches hand-calculated values.

    Given:
      - 2 trials, categories [1, 2, 1, 2], LPP [2.0, 4.0, 6.0, 8.0].
      - Bin edges [2.0, 5.0, 8.0] → 2 bins: [2, 5) and [5, 8].
      - Category 1 at positions 1, 3 (LPP 2.0 in bin 0, LPP 6.0 in bin 1).
      - Trial 1 recalls positions 1, 2 → cat-1 position 1 recalled.
      - Trial 2 recalls positions 3, 4 → cat-1 position 3 recalled.
    When:
      - ``category_lpp_recall_histogram`` is called for category 1.
    Then:
      - Bin 0: 1 recalled / 2 exposures = 0.5.
      - Bin 1: 1 recalled / 2 exposures = 0.5.
    Why this matters:
      - Verifies exact recall-rate computation per LPP bin restricted
        to items of the target category.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0], [3, 4, 0, 0]])
    lpp = jnp.array([[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]])
    categories = jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    bin_edges = jnp.array([2.0, 5.0, 8.0])

    # Act / When
    result = category_lpp_recall_histogram(
        recalls, lpp, categories, 1, bin_edges, 4
    )

    # Assert / Then
    expected = jnp.array([0.5, 0.5])
    assert jnp.allclose(result, expected)


def test_category_lpp_recall_histogram_perfect_recall():
    """Behavior: 100% recall in every bin when all target items recalled.

    Given:
      - Both trials recall positions 1 and 3 (all category-1 items).
    When:
      - ``category_lpp_recall_histogram`` is called for category 1.
    Then:
      - Both bins have recall rate 1.0.
    Why this matters:
      - Perfect recall of all target-category items must produce 1.0
        in every bin, confirming the numerator equals the denominator.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 3, 0, 0], [1, 3, 0, 0]])
    lpp = jnp.array([[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]])
    categories = jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    bin_edges = jnp.array([2.0, 5.0, 8.0])

    # Act / When
    result = category_lpp_recall_histogram(
        recalls, lpp, categories, 1, bin_edges, 4
    )

    # Assert / Then
    expected = jnp.array([1.0, 1.0])
    assert jnp.allclose(result, expected)


def test_cat_recall_by_lpp_matches_histogram_with_explicit_edges():
    """Behavior: Wrapper produces same values as the underlying histogram.

    Given:
      - A dataset with LPP values and category labels.
      - Explicit bin edges matching the histogram test.
    When:
      - ``cat_recall_by_lpp`` is called with those bin edges.
    Then:
      - Result = [0.5, 0.5], matching the histogram function.
    Why this matters:
      - Verifies that the dataset wrapper correctly extracts fields and
        delegates to ``category_lpp_recall_histogram``.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0], [3, 4, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
        "LateLPP": jnp.array([[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]]),
    }
    bin_edges = jnp.array([2.0, 5.0, 8.0])

    # Act / When
    result = cat_recall_by_lpp(
        dataset,
        category_field="condition",
        category_value=1,
        lpp_field="LateLPP",
        bin_edges=bin_edges,
    )

    # Assert / Then
    expected = jnp.array([0.5, 0.5])
    assert jnp.allclose(result, expected)
