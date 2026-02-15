from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.conditional_corec_by_cat import conditional_corec_by_cat
from jaxcmr.helpers import make_dataset


def test_same_cat_one_diff_cat_zero_when_only_target_recalled():
    """Behavior: same-cat rate is 1.0 and diff-cat rate is 0.0 when only
    items of the target category are recalled.

    Given:
      - Two trials recalling positions 1 and 3 only.
      - Categories [1, 2, 1, 2] with ``category_value=1``.
    When:
      - ``conditional_corec_by_cat`` is called.
    Then:
      - result[0] (same-cat) = 1.0: both category-1 anchors see each other
        as recalled same-category neighbors.
      - result[1] (diff-cat) = 0.0: neither category-2 item was recalled,
        so no different-category neighbor of any anchor was recalled.
    Why this matters:
      - Verifies that same-category and different-category co-recall
        rates are computed independently with correct anchor logic.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 3, 0, 0], [1, 3, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
    }

    # Act / When
    result = conditional_corec_by_cat(dataset, category_field="condition", category_value=1)

    # Assert / Then
    expected = jnp.array([1.0, 0.0])
    assert jnp.allclose(result, expected, atol=0.01, equal_nan=True)


def test_exact_rates_with_partial_diff_category_recall():
    """Behavior: correct rates when one diff-category item is recalled.

    Given:
      - Two trials recalling positions 1, 2, 3 (not 4).
      - Categories [1, 2, 1, 2] with ``category_value=1``.
    When:
      - ``conditional_corec_by_cat`` is called.
    Then:
      - result[0] (same-cat) = 1.0: both cat-1 anchors (pos 0, 2) see
        the other as recalled.
      - result[1] (diff-cat) = 0.5: each cat-1 anchor has 2 diff-cat
        neighbors (pos 1 and 3). Pos 1 is recalled; pos 3 is not.
        Total: 2 actual out of 4 possible = 0.5.
    Why this matters:
      - Verifies correct fractional rates when only some different-category
        neighbors are recalled.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 3, 0], [1, 2, 3, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
    }

    # Act / When
    result = conditional_corec_by_cat(dataset, category_field="condition", category_value=1)

    # Assert / Then
    expected = jnp.array([1.0, 0.5])
    assert jnp.allclose(result, expected, atol=0.01, equal_nan=True)
