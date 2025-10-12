import jax.numpy as jnp

from jaxcmr.analyses import nth_item_recall
from jaxcmr.typing import RecallDataset


def _make_dataset(recalls: jnp.ndarray, presentations: jnp.ndarray) -> RecallDataset:
    recalls_arr = jnp.asarray(recalls, dtype=jnp.int32)
    pres_arr = jnp.asarray(presentations, dtype=jnp.int32)
    n_trials = recalls_arr.shape[0]
    list_length = pres_arr.shape[1]
    return {
        "subject": jnp.ones((n_trials, 1), dtype=jnp.int32),
        "listLength": jnp.full((n_trials, 1), list_length, dtype=jnp.int32),
        "pres_itemnos": pres_arr,
        "recalls": recalls_arr,
    }  # type: ignore


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
    dataset = _make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        presentations=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.simple_nth_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.5, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_returns_nan_when_query_not_uniquely_forced():
    """Behavior: ``conditional_nth_item_recall_curve`` yields NaN absent a unique target.

    Given:
      - Trials where other studied items remain unrecalled whenever the query item is still available.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - Each output slot returns NaN because the conditioning set is empty.
    Why this matters:
      - Confirms the analysis refuses to report probabilities for under-constrained states.
    """
    # Arrange / Given
    dataset = _make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        presentations=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    assert jnp.isnan(curve).all().item()


def test_returns_unity_when_query_last_remaining():
    """Behavior: Conditional curve equals 1.0 once every other item is recalled.

    Given:
      - Trials where the query item is the final unrecalled study position and retrieval continues.
    When:
      - ``conditional_nth_item_recall_curve`` is evaluated.
    Then:
      - The final recall position reports probability 1.0; earlier slots remain NaN.
    Why this matters:
      - Ensures the conditioning forces the query item to be recalled when it is the sole option.
    """
    # Arrange / Given
    dataset = _make_dataset(
        recalls=jnp.array([[2, 3, 4, 1], [4, 3, 2, 1]]),
        presentations=jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
    )

    # Act / When
    curve = nth_item_recall.conditional_nth_item_recall_curve(dataset)

    # Assert / Then
    assert jnp.isnan(curve[:3]).all().item()
    assert jnp.allclose(curve[3], jnp.array(1.0)).item()
