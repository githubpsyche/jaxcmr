import jax.numpy as jnp
from jaxcmr.analyses.order_error_rate import (
    order_error_rate,
    trial_order_error_rate,
)


def test_identifies_order_errors_when_items_recalled_in_wrong_positions():
    """Behavior: Flag study positions with misordered recalls.

    Given:
      - A study list and a recall list where two items are swapped.
    When:
      - ``trial_order_error_rate`` is computed.
    Then:
      - Only the swapped positions are marked as order errors.
    Why this matters:
      - Ensures correct identification of misordered recalls.
    """
    # Arrange / Given
    recalls = jnp.array([2, 1, 3], dtype=jnp.int32)
    presentations = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Act / When
    errors = trial_order_error_rate(recalls, presentations)

    # Assert / Then
    assert errors.tolist() == [True, True, False]


def test_computes_mean_order_error_rate_when_averaging_across_trials():
    """Behavior: Average order-error rates across trials.

    Given:
      - Two trials, one with misordered recalls and one perfect.
    When:
      - ``order_error_rate`` is calculated.
    Then:
      - The mean error rates reflect the proportion of misordered recalls.
    Why this matters:
      - Validates aggregation across trials.
    """
    # Arrange / Given
    recalls = jnp.array([[2, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    list_length = 3

    # Act / When
    rates = order_error_rate(recalls, presentations, list_length)

    # Assert / Then
    assert jnp.allclose(rates, jnp.array([0.5, 0.5, 0.0])).item()
