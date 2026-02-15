import jax.numpy as jnp

from jaxcmr.analyses.order_error_rate import (
    trial_order_error_rate,
    order_error_rate,
)
from jaxcmr.helpers import make_dataset


def test_trial_flags_swapped_positions_as_order_errors():
    """Behavior: ``trial_order_error_rate`` flags transposed items.

    Given:
      - A presentation list [1, 2, 3, 4] and recalls [2, 1, 3, 4] where study
        positions 1 and 2 have been swapped in recall output.
    When:
      - ``trial_order_error_rate`` is called.
    Then:
      - Output positions 0 and 1 are True (order errors) while positions 2
        and 3 remain False.
    Why this matters:
      - Validates that items recalled at the wrong serial position are detected
        as transposition errors even though they are on the study list.
    """
    # Arrange / Given
    presentations = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = jnp.array([2, 1, 3, 4], dtype=jnp.int32)

    # Act / When
    result = trial_order_error_rate(recalls, presentations)

    # Assert / Then
    expected = jnp.array([True, True, False, False])
    assert jnp.all(result == expected).item()


def test_dataset_zeros_when_all_trials_in_order():
    """Behavior: ``order_error_rate`` returns zero rates for perfectly ordered trials.

    Given:
      - Two trials where recall matches serial position exactly.
    When:
      - ``order_error_rate`` is called on the dataset.
    Then:
      - The mean order error rate is 0.0 at every study position.
    Why this matters:
      - Confirms the dataset-level wrapper correctly aggregates trial results
        and produces a zero baseline when no transpositions occur.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=jnp.int32,
    )
    presentations = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=jnp.int32,
    )
    dataset = make_dataset(recalls, presentations)

    # Act / When
    rates = order_error_rate(dataset)

    # Assert / Then
    expected = jnp.zeros(4)
    assert jnp.allclose(rates, expected).item()


def test_dataset_expected_rate_when_swap_present():
    """Behavior: ``order_error_rate`` returns the correct mean rate for swaps.

    Given:
      - Two trials: the first swaps positions 1 and 2 ([2, 1, 3, 4]), while
        the second recalls in perfect serial order ([1, 2, 3, 4]).
    When:
      - ``order_error_rate`` is called on the dataset.
    Then:
      - Study positions 0 and 1 each have a mean order error rate of 0.5
        (one error across two trials) and positions 2 and 3 are 0.0.
    Why this matters:
      - Validates that the dataset-level function correctly averages per-trial
        order error flags to produce the expected transposition rate.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [2, 1, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=jnp.int32,
    )
    presentations = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=jnp.int32,
    )
    dataset = make_dataset(recalls, presentations)

    # Act / When
    rates = order_error_rate(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.5, 0.0, 0.0])
    assert jnp.allclose(rates, expected).item()
