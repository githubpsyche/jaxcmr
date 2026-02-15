import jax.numpy as jnp

from jaxcmr.analyses.conditional_corec_by_lag import conditional_corec_by_lag
from jaxcmr.analyses.joint_corec_by_lag import corec_by_lag
from jaxcmr.helpers import make_dataset


def test_conditional_corec_near_one_when_all_items_recalled():
    """Behavior: all lags have conditional co-recall near 1.0 when every item is recalled.

    Given:
      - A dataset where every study position is recalled on every trial.
    When:
      - ``conditional_corec_by_lag`` is computed.
    Then:
      - Values at all positive lags are close to 1.0.
    Why this matters:
      - When every item is recalled, the conditional probability that an
        item at any lag was also recalled must be 1.0.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    dataset = make_dataset(recalls)

    # Act / When
    result = conditional_corec_by_lag(dataset)

    # Assert / Then
    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.ones(3), equal_nan=True)


def test_conditional_corec_zero_when_single_item_recalled():
    """Behavior: co-recall is 0.0 when only one item is recalled per trial.

    Given:
      - A dataset where only position 1 is recalled on each trial.
    When:
      - ``conditional_corec_by_lag`` is computed.
    Then:
      - All positive-lag values are 0.0.
    Why this matters:
      - With only one item recalled, no neighbor at any lag can also be
        recalled, so the conditional co-recall must be zero everywhere.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    dataset = make_dataset(recalls)

    # Act / When
    result = conditional_corec_by_lag(dataset)

    # Assert / Then
    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.zeros(3), equal_nan=True)


def test_joint_corec_exact_values_for_partial_recall():
    """Behavior: joint co-recall matches autocorrelation-based hand calculation.

    Given:
      - Two trials where only positions 1 and 3 are recalled.
    When:
      - ``corec_by_lag`` is computed.
    Then:
      - CoRec(lag=1) = 0.0 because no adjacent pair is co-recalled.
      - CoRec(lag=2) = 0.5 because pair (1,3) is co-recalled but (2,4) is not.
      - CoRec(lag=3) = 0.0 because pair (1,4) is not co-recalled.
    Why this matters:
      - Verifies that the autocorrelation-based co-recall produces correct
        rates for a known recall pattern, not just positivity.
    """
    # Arrange / Given
    # recall mask = [1, 0, 1, 0]
    # correlate([1,0,1,0], [1,0,1,0], "full") = [0, 1, 0, 2, 0, 1, 0]
    # actual_pairs (positive lags) = corr[4:] = [0, 1, 0]
    # possible_pairs = 4 - [1, 2, 3] = [3, 2, 1]
    # CoRec = [0/3, 1/2, 0/1] = [0.0, 0.5, 0.0]
    recalls = jnp.array([[1, 3, 0, 0], [1, 3, 0, 0]])
    dataset = make_dataset(recalls)

    # Act / When
    result = corec_by_lag(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.5, 0.0])
    assert jnp.allclose(result, expected, equal_nan=True)


def test_joint_corec_zero_when_no_items_recalled():
    """Behavior: joint co-recall is 0.0 when no items are recalled.

    Given:
      - A dataset where all recall entries are zero (no recalls).
    When:
      - ``corec_by_lag`` is computed.
    Then:
      - All positive-lag values are 0.0.
    Why this matters:
      - With no recalls, no pair of items can be co-recalled, so the
        joint probability must be zero at every lag.
    """
    # Arrange / Given
    recalls = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    dataset = make_dataset(recalls)

    # Act / When
    result = corec_by_lag(dataset)

    # Assert / Then
    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.zeros(3), equal_nan=True)
