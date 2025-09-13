import matplotlib
matplotlib.use("Agg", force=True)
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import srac
from jaxcmr.typing import RecallDataset


def test_identifies_recalled_positions_when_single_trial():
    """Behavior: Identify recalled study positions.

    Given:
      - Recalled item indices and matching presentations.
      - A maximum study size of three.
    When:
      - ``trial_srac`` is invoked.
    Then:
      - A boolean array flags each correctly recalled position.
    Why this matters:
      - Establishes the per-trial basis for SRAC.
    """
    # Arrange / Given
    recalls = jnp.array([3, 0, 0], dtype=jnp.int32)
    presentations = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Act / When
    result = srac.trial_srac(recalls, presentations, size=1)

    # Assert / Then
    expected = jnp.array([False, False, True])
    assert jnp.array_equal(result, expected).item()


def test_averages_accuracy_when_multiple_trials():
    """Behavior: Average position-specific accuracy across trials.

    Given:
      - Two trials with different recall outcomes.
    When:
      - ``srac`` is computed.
    Then:
      - Mean accuracy per study position is returned.
    Why this matters:
      - Validates aggregate accuracy calculation.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0], [1, 0, 3]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)

    # Act / When
    result = srac.srac(recalls, presentations, list_length=3)

    # Assert / Then
    expected = jnp.array([1.0, 0.5, 0.5])
    assert jnp.allclose(result, expected).item()


def test_returns_axes_when_plotting_srac():
    """Behavior: Return a Matplotlib Axes.

    Given:
      - A minimal ``RecallDataset`` and trial mask.
    When:
      - ``plot_srac`` is called.
    Then:
      - A Matplotlib ``Axes`` with a ``Figure`` is returned.
    Why this matters:
      - Confirms visualization interface.
    """
    # Arrange / Given
    dataset: RecallDataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)

    # Act / When
    axis = srac.plot_srac(dataset, trial_mask)

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
