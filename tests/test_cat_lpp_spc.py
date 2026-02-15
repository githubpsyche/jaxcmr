from typing import Any

import jax.numpy as jnp

from jaxcmr.analyses.cat_lpp_spc import category_lpp_values, cat_lpp_spc
from jaxcmr.helpers import make_dataset


def test_category_lpp_values_returns_filtered_means():
    """Behavior: mean LPP is computed only at positions matching the category.

    Given:
      - LPP values and alternating category labels across two trials.
    When:
      - ``category_lpp_values`` is called for the matching category.
    Then:
      - Returned means reflect only the matching-category positions.
    Why this matters:
      - Validates that category filtering correctly restricts the LPP
        average to items of interest while ignoring other categories.
    """
    # Arrange / Given
    lpp = jnp.array([[5.0, 3.0, 4.0, 2.0], [6.0, 4.0, 5.0, 3.0]])
    categories = jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    category_value = 1

    # Act / When
    result = category_lpp_values(lpp, categories, category_value)

    # Assert / Then
    assert result.shape == (4,)
    # Position 0: both trials are category 1 -> mean of (5.0, 6.0) = 5.5
    assert jnp.allclose(result[0], 5.5)
    # Position 2: both trials are category 1 -> mean of (4.0, 5.0) = 4.5
    assert jnp.allclose(result[2], 4.5)


def test_category_lpp_values_nan_or_inf_when_no_matches():
    """Behavior: division by zero produces NaN or inf when no items match.

    Given:
      - LPP values where all items belong to category 1.
    When:
      - ``category_lpp_values`` is called for category 2 (absent).
    Then:
      - Results contain NaN or inf from dividing by zero counts.
    Why this matters:
      - Confirms expected behavior when the requested category has no
        items at any position (denominator is zero).
    """
    # Arrange / Given
    lpp = jnp.array([[5.0, 3.0, 4.0, 2.0], [6.0, 4.0, 5.0, 3.0]])
    categories = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    category_value = 2

    # Act / When
    result = category_lpp_values(lpp, categories, category_value)

    # Assert / Then
    assert result.shape == (4,)
    assert jnp.all(jnp.isnan(result) | jnp.isinf(result))


def test_cat_lpp_spc_exact_values():
    """Behavior: dataset-level LPP SPC matches hand-calculated means.

    Given:
      - Two trials with LPP values and categories [1, 2, 1, 2].
      - Category 1 at positions 0 (LPP 5.0, 6.0) and 2 (LPP 4.0, 5.0).
    When:
      - ``cat_lpp_spc`` is called for category 1.
    Then:
      - Position 0: mean LPP = (5+6)/2 = 5.5.
      - Position 2: mean LPP = (4+5)/2 = 4.5.
      - Positions 1, 3: NaN or inf (no category-1 items).
    Why this matters:
      - Verifies the wrapper produces the same values as the underlying
        ``category_lpp_values``.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0, 0], [3, 4, 0, 0]])
    dataset: Any = {
        **make_dataset(recalls),
        "condition": jnp.array([[1, 2, 1, 2], [1, 2, 1, 2]]),
        "LateLPP": jnp.array([[5.0, 3.0, 4.0, 2.0], [6.0, 4.0, 5.0, 3.0]]),
    }

    # Act / When
    result = cat_lpp_spc(dataset, category_field="condition", category_value=1, lpp_field="LateLPP")

    # Assert / Then
    assert result.shape == (4,)
    assert jnp.isclose(result[0], 5.5)
    assert jnp.isclose(result[2], 4.5)
