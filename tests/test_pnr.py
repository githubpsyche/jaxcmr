import pytest

jnp = pytest.importorskip("jax.numpy")

from jaxcmr.analyses.pnr import (
    fixed_pres_pnr,
    available_recalls,
    conditional_fixed_pres_pnr,
    pnr,
)


def test_probability_vector_when_first_recall_analyzed():
    """Behavior: compute probability per study position.

    Given:
      - first recall from multiple trials
    When:
      - calculating fixed-presence PNR
    Then:
      - probabilities match observed counts
    Why this matters:
      - requirement: verifies basic PNR calculation
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0], [2, 1, 0]])

    # Act / When
    result = fixed_pres_pnr(recalls, list_length=3)

    # Assert / Then
    assert jnp.allclose(result, jnp.array([0.5, 0.5, 0.0]))


def test_availability_mask_when_prior_items_recalled():
    """Behavior: mark prior recalled positions unavailable.

    Given:
      - recall sequence with earlier items
    When:
      - computing availability mask at later recall
    Then:
      - positions of earlier recalls are False
    Why this matters:
      - invariant: prevents item repetition
    """
    # Arrange / Given
    recalls = jnp.array([1, 0, 0])

    # Act / When
    mask = available_recalls(recalls, query_recall_position=1, list_length=3)

    # Assert / Then
    assert jnp.array_equal(mask, jnp.array([False, True, True]))


def test_probability_zero_when_position_never_available():
    """Behavior: return 0 when study position unavailable.

    Given:
      - trials where a study position is always recalled earlier
    When:
      - computing conditional PNR at later recall
    Then:
      - probability for that position is 0
    Why this matters:
      - invariant: avoids divide-by-zero
    """
    # Arrange / Given
    recalls = jnp.array([[2, 1], [2, 0]])

    # Act / When
    result = conditional_fixed_pres_pnr(recalls, list_length=2, query_recall_position=1)

    # Assert / Then
    assert result[1] == 0.0


def test_repetition_handling_when_items_repeat():
    """Behavior: distribute recall probability across repeated study positions.

    Given:
      - study lists with repeated items
    When:
      - calculating PNR
    Then:
      - repeated positions share probability mass
    Why this matters:
      - requirement: supports repeated presentations
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0], [1, 0]])
    presentations = jnp.array([[1, 2, 1], [1, 2, 1]])

    # Act / When
    result = pnr(recalls, presentations, list_length=3, size=2)

    # Assert / Then
    assert jnp.allclose(result, jnp.array([1.0, 0.0, 1.0]))

