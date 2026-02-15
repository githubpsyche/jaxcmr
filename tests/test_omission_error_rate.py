import jax.numpy as jnp

from jaxcmr.analyses.omission_error_rate import (
    trial_omission_error_rate,
    omission_error_rate,
)
from jaxcmr.helpers import make_dataset


def test_trial_flags_omission_when_position_not_recalled():
    """Behavior: ``trial_omission_error_rate`` flags unrecalled study positions.

    Given:
      - A presentation list [1, 2, 3, 4] and recalls [1, 0, 0, 0] where only
        the item at study position 1 is recalled.
    When:
      - ``trial_omission_error_rate`` is called.
    Then:
      - Study positions 2, 3, and 4 are True (omitted) and position 1 is False.
    Why this matters:
      - Validates that positions whose items were never recalled are correctly
        identified as omission errors.
    """
    # Arrange / Given
    presentations = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = jnp.array([1, 0, 0, 0], dtype=jnp.int32)

    # Act / When
    result = trial_omission_error_rate(recalls, presentations)

    # Assert / Then
    expected = jnp.array([False, True, True, True])
    assert jnp.all(result == expected).item()


def test_dataset_zeros_when_all_trials_perfect():
    """Behavior: ``omission_error_rate`` returns zero rates for perfect recall.

    Given:
      - Two trials where every study position is recalled.
    When:
      - ``omission_error_rate`` is called on the dataset.
    Then:
      - The mean omission rate is 0.0 at every study position.
    Why this matters:
      - Confirms the dataset-level wrapper correctly aggregates trial results
        and produces a zero baseline when recall is complete.
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
    rates = omission_error_rate(dataset)

    # Assert / Then
    expected = jnp.zeros(4)
    assert jnp.allclose(rates, expected).item()


def test_dataset_half_rate_when_half_positions_missed():
    """Behavior: ``omission_error_rate`` returns 0.5 at missed positions.

    Given:
      - Two trials with list length 4: the first recalls positions 1 and 2 only,
        the second recalls positions 3 and 4 only.
    When:
      - ``omission_error_rate`` is called on the dataset.
    Then:
      - Every study position has a mean omission rate of 0.5 because each
        position is recalled in exactly one of the two trials.
    Why this matters:
      - Validates that the dataset-level function correctly averages per-trial
        omission flags, producing the expected rate for partial recall.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
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
    rates = omission_error_rate(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.5, 0.5, 0.5])
    assert jnp.allclose(rates, expected).item()
