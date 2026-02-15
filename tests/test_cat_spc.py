from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.cat_spc import category_recall_counts, cat_spc
from jaxcmr.helpers import make_dataset


def test_category_recall_counts_exact_values():
    """Behavior: exact recall counts at matching positions, zero elsewhere.

    Given:
      - Recalls [1, 2, 3, 0] with categories [1, 2, 1, 2].
    When:
      - ``category_recall_counts`` is called for category 1.
    Then:
      - Result is [1, 0, 1, 0]: positions 0 and 2 match category 1 and
        were recalled; positions 1 and 3 are category 2.
    Why this matters:
      - Verifies the exact intersection of bincount-based recall detection
        and category masking, not just positivity.
    """
    # Arrange / Given
    recalls = jnp.array([1, 2, 3, 0], dtype=jnp.int32)
    categories = jnp.array([1, 2, 1, 2], dtype=jnp.int32)

    # Act / When
    result = category_recall_counts(recalls, categories, 1, 4)

    # Assert / Then
    expected = jnp.array([1, 0, 1, 0])
    assert jnp.array_equal(result, expected)


def test_category_recall_counts_zero_when_category_absent():
    """Behavior: all zeros when no items match the target category.

    Given:
      - Recalls from a trial where all items belong to category 1.
    When:
      - ``category_recall_counts`` is called for category 2 (absent).
    Then:
      - All counts are zero.
    Why this matters:
      - Ensures that filtering by a non-matching category produces
        an empty count vector.
    """
    # Arrange / Given
    recalls = jnp.array([1, 2, 3, 0], dtype=jnp.int32)
    categories = jnp.array([1, 1, 1, 1], dtype=jnp.int32)

    # Act / When
    result = category_recall_counts(recalls, categories, 2, 4)

    # Assert / Then
    assert jnp.allclose(result, jnp.zeros(4))


def test_cat_spc_exact_values():
    """Behavior: category SPC matches hand-calculated recall rates.

    Given:
      - Trial 1 recalls position 1 (cat 1, recalled) and position 2 (cat 2).
      - Trial 2 recalls position 3 (cat 1, recalled) and position 4 (cat 2).
      - Categories [1, 2, 1, 2] on both trials.
    When:
      - ``cat_spc`` is called for category 1.
    Then:
      - Position 0: 1 trial recalled / 1 cat-1 exposure per trial = 0.5 overall.
      - Position 2: 1 trial recalled / 1 cat-1 exposure per trial = 0.5 overall.
      - Positions 1, 3: not category 1, so NaN or 0.
    Why this matters:
      - Verifies exact probability computation, not just range validity.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0, 0, 0], [3, 0, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
    }

    # Act / When
    result = cat_spc(dataset, category_field="condition", category_value=1)

    # Assert / Then
    assert result.shape == (4,)
    # Position 0: recalled in trial 1 only → 1/2 = 0.5
    assert jnp.isclose(result[0], 0.5)
    # Position 2: recalled in trial 2 only → 1/2 = 0.5
    assert jnp.isclose(result[2], 0.5)
