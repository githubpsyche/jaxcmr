import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import spc
from jaxcmr.typing import RecallDataset


def test_returns_recall_rates_when_lists_are_uniform():
    """Behavior: ``fixed_pres_spc`` computes recall rates by study position.

    Given:
      - Uniform study lists with recalled items.
    When:
      - ``fixed_pres_spc`` is called.
    Then:
      - Recall rates per position are returned.
    Why this matters:
      - Ensures baseline serial position computation.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32)
    list_length = 3

    # Act / When
    rates = spc.fixed_pres_spc(recalls, list_length)

    # Assert / Then
    assert jnp.allclose(rates, jnp.array([0.5, 1.0, 0.5]))


def test_returns_recall_rates_when_using_presentations():
    """Behavior: ``spc`` accounts for study presentations.

    Given:
      - Recall and presentation matrices.
    When:
      - ``spc`` is called.
    Then:
      - Recall rates per study position are returned.
    Why this matters:
      - Validates serial position computation with presentations.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    list_length = 3

    # Act / When
    rates = spc.spc(recalls, presentations, list_length, size=1)

    # Assert / Then
    assert jnp.allclose(rates, jnp.array([0.5, 1.0, 0.5]))


def test_returns_axes_when_plotting_spc():
    """Behavior: ``plot_spc`` provides a Matplotlib ``Axes``.

    Given:
      - A minimal ``RecallDataset``.
    When:
      - ``plot_spc`` is called.
    Then:
      - A Matplotlib ``Axes`` with a ``Figure`` is returned.
    Why this matters:
      - Ensures visualization utility returns standard objects.
    """
    # Arrange / Given
    dataset: RecallDataset = {
        "subject": jnp.array([[1], [1]], dtype=jnp.int32),
        "listLength": jnp.array([[3], [3]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        "recalls": jnp.array([[1, 2, 0], [2, 3, 0]], dtype=jnp.int32),
        "pres_itemids": jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
    }
    trial_mask = jnp.array([True, True], dtype=bool)

    # Act / When
    axis = spc.plot_spc(dataset, trial_mask, size=1)

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)
