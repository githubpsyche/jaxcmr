import jax.numpy as jnp
import pytest
from jaxcmr.analyses import intrusion_error_rate as ier


@pytest.mark.parametrize(
    "recalls,presentations,expected",
    [
        (jnp.array([4, 1, 2], dtype=jnp.int32), jnp.array([1, 2, 3], dtype=jnp.int32), [True, False, False]),
        (jnp.array([1, 4, 2], dtype=jnp.int32), jnp.array([1, 2, 3], dtype=jnp.int32), [False, True, False]),
    ],
)
def test_flags_intrusions_when_item_not_studied(recalls, presentations, expected):
    """Behavior: Identify intrusions only for items absent from the study list.

    Given:
      - Recall and study lists for a single trial.
    When:
      - ``trial_intrusion_error_rate`` is invoked.
    Then:
      - Positions with nonstudied recalls are flagged as intrusions.
    Why this matters:
      - Ensures correct classification of intrusion errors.
    """
    # Arrange / Given
    # Parameters supply the setup

    # Act / When
    result = ier.trial_intrusion_error_rate(recalls, presentations)

    # Assert / Then
    assert jnp.array_equal(result, jnp.array(expected, dtype=bool)).item()


@pytest.mark.parametrize(
    "recalls,presentations,expected",
    [
        (jnp.array([0, 1, 0], dtype=jnp.int32), jnp.array([1, 2, 3], dtype=jnp.int32), [False, False, False]),
        (jnp.array([2, 0, 3], dtype=jnp.int32), jnp.array([1, 2, 0], dtype=jnp.int32), [False, False, False]),
    ],
)
def test_returns_false_when_recall_missing_or_in_study(recalls, presentations, expected):
    """Behavior: Treat studied or missing recalls as non-intrusions.

    Given:
      - Recall and study lists for a single trial.
    When:
      - ``trial_intrusion_error_rate`` is invoked.
    Then:
      - No positions are flagged as intrusions.
    Why this matters:
      - Avoids false positives from studied items or padding.
    """
    # Arrange / Given
    # Parameters supply the setup

    # Act / When
    result = ier.trial_intrusion_error_rate(recalls, presentations)

    # Assert / Then
    assert jnp.array_equal(result, jnp.array(expected, dtype=bool)).item()


def test_averages_intrusion_rates_when_multiple_trials():
    """Behavior: Average intrusion flags across trials.

    Given:
      - Recall and study lists for multiple trials.
    When:
      - ``intrusion_error_rate`` is called.
    Then:
      - The mean intrusion rate per position is returned.
    Why this matters:
      - Aggregates trial-level intrusions into a summary metric.
    """
    # Arrange / Given
    recalls = jnp.array([[4, 1, 2], [1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)

    # Act / When
    rates = ier.intrusion_error_rate(recalls, presentations, list_length=3)

    # Assert / Then
    assert jnp.allclose(rates, jnp.array([0.5, 0.0, 0.0])).item()
