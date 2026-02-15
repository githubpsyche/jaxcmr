import jax.numpy as jnp

from jaxcmr.analyses.intrusion_error_rate import (
    trial_intrusion_error_rate,
    intrusion_error_rate,
)
from jaxcmr.helpers import make_dataset


def test_trial_flags_intrusion_when_recall_not_on_list():
    """Behavior: ``trial_intrusion_error_rate`` flags extra-list items.

    Given:
      - A presentation list [1, 2, 3, 4] and recalls [1, 99, 2, 0] where item
        99 does not appear on the study list.
    When:
      - ``trial_intrusion_error_rate`` is called.
    Then:
      - Position 1 is True (intrusion) and all other positions are False.
    Why this matters:
      - Validates that the function detects an item recalled from outside the
        current study list as an intrusion error.
    """
    # Arrange / Given
    presentations = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    recalls = jnp.array([1, 99, 2, 0], dtype=jnp.int32)

    # Act / When
    result = trial_intrusion_error_rate(recalls, presentations)

    # Assert / Then
    expected = jnp.array([False, True, False, False])
    assert jnp.all(result == expected).item()


def test_dataset_zeros_when_all_recalls_correct():
    """Behavior: ``intrusion_error_rate`` returns zero rates for clean trials.

    Given:
      - Two trials where every nonzero recall is an item from the study list.
    When:
      - ``intrusion_error_rate`` is called on the dataset.
    Then:
      - The mean intrusion rate is 0.0 at every study position.
    Why this matters:
      - Confirms the dataset-level wrapper correctly aggregates trial results
        and produces a zero baseline when no intrusions are present.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [1, 3, 2, 0],
            [2, 4, 0, 0],
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
    rates = intrusion_error_rate(dataset)

    # Assert / Then
    expected = jnp.zeros(4)
    assert jnp.allclose(rates, expected).item()


def test_dataset_expected_rate_when_intrusion_present():
    """Behavior: ``intrusion_error_rate`` returns the correct mean rate.

    Given:
      - Two trials: the first has an intrusion at output position 1 (item 99),
        the second has no intrusions. Each trial has list length 4.
    When:
      - ``intrusion_error_rate`` is called on the dataset.
    Then:
      - The mean intrusion rate at position 1 is 0.5 and 0.0 elsewhere.
    Why this matters:
      - Validates that the dataset-level function correctly averages per-trial
        intrusion flags across trials.
    """
    # Arrange / Given
    recalls = jnp.array(
        [
            [1, 99, 2, 0],
            [1, 3, 2, 0],
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
    rates = intrusion_error_rate(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.5, 0.0, 0.0])
    assert jnp.allclose(rates, expected).item()
