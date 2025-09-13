import jax.numpy as jnp
from jaxcmr.analyses import rpl


def test_marks_bin_as_present_and_recalled_when_item_repeats():
    """Behavior: Return one-hot lag vectors for presented and recalled bins.

    Given:
      - A presentation with a repeated target item.
      - A recall list containing the item's first position.
    When:
      - ``item_lag_counts`` is invoked.
    Then:
      - Presented and recalled vectors flag the correct lag bin.
    Why this matters:
      - Supports accurate aggregation of repetition effects.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 1, 3], dtype=jnp.int32)
    recalls = jnp.array([1, 0, 0], dtype=jnp.int32)
    n_bins = 4
    max_lag = 2

    # Act / When
    presented, recalled = rpl.item_lag_counts(1, recalls, presentation, max_lag, n_bins)

    # Assert / Then
    expected = jnp.array([0, 0, 1, 0], dtype=bool)
    assert jnp.array_equal(presented, expected).item()
    assert jnp.array_equal(recalled, expected).item()


def test_reports_unity_probability_when_only_repeated_item_recalled():
    """Behavior: Tabulate recall probability across lag bins.

    Given:
      - A single trial with one repeated item.
    When:
      - ``recall_probability_by_lag`` is invoked.
    Then:
      - Only the lag bin of the repeated item has probability one.
    Why this matters:
      - Confirms correct classification of repeats.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 0, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 1, 3]], dtype=jnp.int32)

    # Act / When
    probs = rpl.recall_probability_by_lag(
        recalls, presentations, list_length=3, max_lag=2
    )

    # Assert / Then
    expected = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(probs, expected).item()
