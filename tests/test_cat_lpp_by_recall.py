from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.cat_lpp_by_recall import expand_categories_by_recall
from jaxcmr.helpers import make_dataset


def test_expand_categories_exact_remapping():
    """Behavior: recalled items get ``cat*2``, non-recalled get ``cat*2-1``.

    Given:
      - One trial recalling positions 1 and 3. Categories are [1, 1, 2, 2].
    When:
      - ``expand_categories_by_recall`` is called.
    Then:
      - Position 0 (cat 1, recalled): 1*2-1+1 = 2.
      - Position 1 (cat 1, not recalled): 1*2-1+0 = 1.
      - Position 2 (cat 2, recalled): 2*2-1+1 = 4.
      - Position 3 (cat 2, not recalled): 2*2-1+0 = 3.
    Why this matters:
      - Verifies the exact arithmetic mapping so downstream LPP analyses
        correctly separate recalled vs non-recalled items by category.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 3, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 1, 2, 2]]),
    }

    # Act / When
    result = expand_categories_by_recall(dataset, category_field="condition")

    # Assert / Then
    expected = jnp.array([[2, 1, 4, 3]], dtype=jnp.int32)
    assert jnp.array_equal(result, expected)


def test_expand_categories_consistent_across_trials():
    """Behavior: identical recall patterns produce identical remapped labels.

    Given:
      - Two trials with the same recalls [1, 2, 0, 0] and categories [1, 1, 1, 1].
    When:
      - ``expand_categories_by_recall`` is called.
    Then:
      - Both trials have the same remapped labels: [2, 2, 1, 1] because
        positions 0 and 1 are recalled (cat*2=2) and positions 2 and 3
        are not recalled (cat*2-1=1).
    Why this matters:
      - Non-recalled items must retain a stable category identity so
        that their LPP values can be aggregated correctly.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0], [1, 2, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
    }

    # Act / When
    result = expand_categories_by_recall(dataset, category_field="condition")

    # Assert / Then
    expected = jnp.array([[2, 2, 1, 1], [2, 2, 1, 1]], dtype=jnp.int32)
    assert jnp.array_equal(result, expected)
