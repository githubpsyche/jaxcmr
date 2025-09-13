import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import backrepcrp
from jaxcmr.typing import RecallDataset


def test_retains_state_when_tabulating_zero_recall():
    """Behavior: ``tabulate(0)`` leaves lag counts unchanged.

    Given:
      - A ``RepCRPTabulation`` instance.
    When:
      - ``tabulate`` is called with ``0``.
    Then:
      - Actual and available lag counts are unchanged.
    Why this matters:
      - Maintains sentinel semantics for no-op transitions.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    tab = backrepcrp.RepCRPTabulation(presentation, first_recall=1, min_lag=1, size=2)
    original_actual = tab.actual_lags.copy()
    original_avail = tab.avail_lags.copy()

    # Act / When
    result = tab.tabulate(0)

    # Assert / Then
    assert jnp.array_equal(result.actual_lags, original_actual).item()
    assert jnp.array_equal(result.avail_lags, original_avail).item()


def test_reports_true_when_recall_has_spaced_repetitions():
    """Behavior: ``should_tabulate`` identifies spaced repetitions.

    Given:
      - An item repeated at two study positions with sufficient spacing.
    When:
      - ``should_tabulate`` is queried for that item.
    Then:
      - ``True`` is returned.
    Why this matters:
      - Ensures tabulation only for well-separated repeats.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3, 1], dtype=jnp.int32)
    tab = backrepcrp.RepCRPTabulation(presentation, first_recall=2, min_lag=2, size=2)

    # Act / When
    flag = tab.should_tabulate(1)

    # Assert / Then
    assert flag is True


def test_returns_nan_when_no_valid_transitions():
    """Behavior: ``repcrp`` yields ``NaN`` without valid transitions.

    Given:
      - Trials without repeated items meeting ``min_lag``.
    When:
      - ``repcrp`` is computed.
    Then:
      - The result contains ``NaN`` values.
    Why this matters:
      - Signals lack of tabulatable transitions.
    """
    # Arrange / Given
    trials = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    # Act / When
    out = backrepcrp.repcrp(trials, presentations, list_length=3, min_lag=4, size=2)

    # Assert / Then
    assert jnp.isnan(out).all().item()


def test_returns_axes_when_plotting_back_rep_crp():
    """Behavior: ``plot_back_rep_crp`` returns a Matplotlib ``Axes``.

    Given:
      - A minimal ``RecallDataset`` with repeated items.
    When:
      - ``plot_back_rep_crp`` is invoked.
    Then:
      - A Matplotlib ``Axes`` and ``Figure`` are produced.
    Why this matters:
      - Confirms plotting API returns standard objects.
    """
    # Arrange / Given
    dataset: RecallDataset = {
        "subject": jnp.array([[1]], dtype=jnp.int32),
        "listLength": jnp.array([[4]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 1, 2]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 1, 2]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 1, 2]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True], dtype=bool)

    # Act / When
    axis = backrepcrp.plot_back_rep_crp(dataset, trial_mask, max_lag=1, min_lag=1, size=2)

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)

