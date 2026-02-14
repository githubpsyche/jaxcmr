import jax.numpy as jnp

from jaxcmr.analyses import nth_item_recall
from jaxcmr.helpers import make_dataset


def test_returns_expected_probabilities_when_using_simple_curve():
    """Behavior: ``simple_nth_item_recall_curve`` averages matches per recall slot.

    Given:
      - Trials where the first studied item appears once at recall position 1 and once at position 2.
    When:
      - ``simple_nth_item_recall_curve`` is computed.
    Then:
      - The resulting probabilities are 0.5 for the first two recall positions and 0.0 afterwards.
    Why this matters:
      - Confirms the baseline calculation before availability gating is applied.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        pres_itemnos=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.simple_nth_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.5, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_reports_probability_until_target_is_recalled():
    """Behavior: ``conditional_nth_item_recall_curve`` tracks probability until recall.

    Given:
      - Trials where the query item first appears at the second output position.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - The first two positions yield finite probabilities and later slots report NaN.
    Why this matters:
      - Confirms conditioning only requires the preceding recall to be non-terminating.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        pres_itemnos=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    assert jnp.allclose(curve[:2], jnp.array([0.5, 1.0])).item()
    assert jnp.isnan(curve[2]).item()


def test_returns_unity_when_query_last_remaining():
    """Behavior: Conditional curve equals 1.0 once every other item is recalled.

    Given:
      - Trials where the query item is the final unrecalled study position and retrieval continues.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - The final recall position reports probability 1.0 with earlier slots evaluating to 0.0.
    Why this matters:
      - Ensures the conditioning captures certainty when the query is the sole option.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=jnp.array([[2, 3, 4, 1], [4, 3, 2, 1]]),
        pres_itemnos=jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.0, 0.0, 1.0])
    assert jnp.allclose(curve, expected, equal_nan=True).item()


def test_handles_positions_not_reached_before_stop():
    """Behavior: Conditional curve drops slots following termination.

    Given:
      - Trials where one sequence terminates before the final recall.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - The final slot contributes zero because the preceding recall was non-zero but the current event is termination.
    Why this matters:
      - Ensures conditioning still counts opportunities even when the outcome is a stop.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=jnp.array([[2, 3, 0], [1, 3, 4]]),
        pres_itemnos=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.0, 0.0])
    assert jnp.allclose(curve, expected, equal_nan=True).item()


def test_ignores_trials_that_stop_before_slot():
    """Behavior: Conditional curve skips slots whose prior recall is termination.

    Given:
      - Trials where the query item is recalled and the next slot is already padding.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - Later slots become NaN because no valid opportunities remain.
    Why this matters:
      - Confirms the denominator ignores exposures that follow a termination.
    """
    # Arrange / Given
    dataset = make_dataset(
        recalls=jnp.array([[1, 0, 0], [2, 1, 3]]),
        pres_itemnos=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    assert jnp.allclose(curve[:2], jnp.array([0.5, 1.0])).item()
    assert jnp.isnan(curve[2]).item()
