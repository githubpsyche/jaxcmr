import jax.numpy as jnp

from jaxcmr.analyses import first_item_recall
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
    }


def test_returns_expected_probabilities_when_using_simple_curve():
    """Behavior: ``simple_first_item_recall_curve`` averages matches per recall slot.

    Given:
      - Trials where the first studied item appears once at recall position 1 and once at position 2.
    When:
      - ``simple_first_item_recall_curve`` is computed.
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
    curve = first_item_recall.simple_first_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.5, 0.5, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_ignores_second_occurrence_when_first_item_already_recalled():
    """Behavior: ``first_item_recall_curve`` ignores later recalls once the first item is used.

    Given:
      - Trials where the first item is recalled at the first output and never appears again.
    When:
      - ``first_item_recall_curve`` is computed.
    Then:
      - Only the first recall position contributes probability; later slots remain zero.
    Why this matters:
      - Validates availability gating prevents double-counting the first item.
    """
    # Arrange / Given
    dataset = _make_dataset(
        recalls=jnp.array([[1, 2, 3], [2, 1, 3]]),
        presentations=jnp.array([[1, 2, 3], [1, 2, 3]]),
    )

    # Act / When
    curve = first_item_recall.first_item_recall_curve(dataset)

    # Assert / Then
    expected = jnp.array([.5, 1.0, 0.0])
    assert jnp.allclose(curve, expected).item()
