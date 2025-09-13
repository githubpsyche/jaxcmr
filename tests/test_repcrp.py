import pytest
import jax.numpy as jnp

from jaxcmr.analyses import repcrp


@pytest.mark.parametrize(
    "presentation,min_lag,expected",
    [
        (jnp.array([1, 0, 0, 0, 1], dtype=jnp.int32), 3, True),
        (jnp.array([1, 2, 1], dtype=jnp.int32), 2, False),
    ],
)
def test_reports_eligibility_when_spacing_changes(presentation, min_lag, expected):
    """Behavior: Report eligibility to tabulate repeated items when spacing threshold changes.

    Given:
      - A presentation list with a repeated item.
      - A minimum lag requirement.
    When:
      - ``RepCRPTabulation`` is initialized and queried.
    Then:
      - ``should_tabulate`` matches the expected eligibility.
    Why this matters:
      - Ensures repetitions contribute only when sufficiently spaced.
    """
    # Arrange / Given handled by parametrization

    # Act / When
    tab = repcrp.RepCRPTabulation(presentation=presentation, first_recall=1, min_lag=min_lag, size=2)

    # Assert / Then
    assert tab.should_tabulate().item() is expected


def test_returns_nan_vector_when_no_repetitions():
    """Behavior: Produce NaNs when no repeated items exist.

    Given:
      - Trials and presentations lacking repeated items.
    When:
      - ``repcrp`` is computed.
    Then:
      - The result consists entirely of ``NaN`` values.
    Why this matters:
      - Signals absence of valid transitions.
    """
    # Arrange / Given
    trials = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    # Act / When
    values = repcrp.repcrp(trials, presentations, list_length=3, size=2, min_lag=1)

    # Assert / Then
    assert jnp.isnan(values).all().item()


def test_outputs_probabilities_within_unit_interval_when_repetition_present():
    """Behavior: Provide probabilities between 0 and 1 for valid repetitions.

    Given:
      - A trial with a repeated item spaced beyond ``min_lag``.
    When:
      - ``repcrp`` is computed.
    Then:
      - Non-``NaN`` entries fall within the unit interval.
    Why this matters:
      - Validates probability semantics of the analysis.
    """
    # Arrange / Given
    trials = jnp.array([[1, 1, 0, 0, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 0, 0, 0, 1]], dtype=jnp.int32)

    # Act / When
    values = repcrp.repcrp(trials, presentations, list_length=5, min_lag=3, size=2)

    # Assert / Then
    finite = ~jnp.isnan(values)
    assert jnp.logical_and(values[finite] >= 0, values[finite] <= 1).all().item()
