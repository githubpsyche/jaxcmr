import jax.numpy as jnp

from jaxcmr.analyses.omission_error_rate import (
    omission_error_rate,
    trial_omission_error_rate,
)


def test_returns_positionwise_omission_rates_when_items_are_not_recalled():
    """Behavior: compute positionwise omission error rates.

    Given:
      - recall and presentation arrays with omissions
    When:
      - calculating omission error rate
    Then:
      - returns omission rates per study position
    Why this matters:
      - verifies correct omission detection
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0], [1, 0, 3]])
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]])

    # Act / When
    rates = omission_error_rate(recalls, presentations, list_length=3)

    # Assert / Then
    assert jnp.allclose(rates, jnp.array([0.0, 0.5, 0.5])).item()


def test_returns_false_for_padded_positions_when_lists_are_unequal():
    """Behavior: ignore padded study positions.

    Given:
      - a study list with padded zero positions
    When:
      - computing omission flags for the trial
    Then:
      - padded positions are flagged as not omitted
    Why this matters:
      - ensures padding does not inflate error rates
    """
    # Arrange / Given
    recalls = jnp.array([1, 2, 0])
    presentations = jnp.array([1, 2, 0])

    # Act / When
    flags = trial_omission_error_rate(recalls, presentations)

    # Assert / Then
    assert flags.tolist() == [False, False, False]

