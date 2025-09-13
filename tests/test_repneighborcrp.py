import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import repneighborcrp
from jaxcmr.typing import RecallDataset


def test_returns_arrays_when_tabulating_trial_with_repeat():
    """Behavior: Provide actual and available lag counts.

    Given:
      - A single recall trial with a repeated item.
      - A presentation containing the repeated item.
    When:
      - ``tabulate_trial`` is executed.
    Then:
      - Two lag vectors of equal length are returned.
    Why this matters:
      - Validates basic tabulation for neighbor repetitions.
    """
    # Arrange / Given
    trial = jnp.array([7, 1, 0, 0], dtype=jnp.int32)
    presentation = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)

    # Act / When
    actual, avail = repneighborcrp.tabulate_trial(
        trial, presentation, direction="j2i", use_lag2=False
    )

    # Assert / Then
    expected_len = 2 * presentation.size - 1
    assert isinstance(actual, jnp.ndarray)
    assert isinstance(avail, jnp.ndarray)
    assert actual.shape == (expected_len,)
    assert avail.shape == (expected_len,)


def test_returns_probabilities_when_analyzing_trials():
    """Behavior: Compute lag-CRP probabilities.

    Given:
      - Trials and presentations with a repeated item.
    When:
      - ``repneighborcrp`` is invoked.
    Then:
      - A probability vector of expected length is returned.
    Why this matters:
      - Confirms end-to-end analysis produces probabilities.
    """
    # Arrange / Given
    trials = jnp.array([[7, 1, 0, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3, 4, 5, 1, 6, 7]], dtype=jnp.int32)

    # Act / When
    probs = repneighborcrp.repneighborcrp(
        trials, presentations, direction="j2i", use_lag2=False
    )

    # Assert / Then
    expected_len = 2 * presentations.shape[1] - 1
    assert isinstance(probs, jnp.ndarray)
    assert probs.shape == (expected_len,)


def test_returns_axes_when_plotting_rep_neighbor_crp():
    """Behavior: ``plot_rep_neighbor_crp`` provides a Matplotlib ``Axes``.

    Given:
      - A minimal ``RecallDataset`` with a repeated item.
    When:
      - ``plot_rep_neighbor_crp`` is called.
    Then:
      - A Matplotlib ``Axes`` with a ``Figure`` is returned.
    Why this matters:
      - Ensures visualization utility returns standard objects.
    """
    # Arrange / Given
    trials = jnp.array([[7, 1, 0, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3, 4, 5, 1, 6, 7]], dtype=jnp.int32)
    dataset: RecallDataset = {
        "subject": jnp.array([[1]], dtype=jnp.int32),
        "listLength": jnp.array([[presentations.shape[1]]], dtype=jnp.int32),
        "pres_itemnos": presentations,
        "recalls": trials,
    }
    trial_mask = jnp.array([True], dtype=bool)

    # Act / When
    axis = repneighborcrp.plot_rep_neighbor_crp(
        dataset, trial_mask, max_lag=1, direction="j2i", use_lag2=False
    )

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)

