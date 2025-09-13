import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses import rspc
from jaxcmr.typing import RecallDataset


def test_flags_all_positions_when_recall_in_forward_order():
    """Behavior: Mark every study position for perfect forward recall.

    Given:
      - A study list and matching forward recalls.
    When:
      - ``tabulate_trial`` is invoked.
    Then:
      - All study positions are flagged ``True``.
    Why this matters:
      - Confirms full accuracy for ideal behavior.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    trial = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Act / When
    flags = rspc.tabulate_trial(trial, presentation)

    # Assert / Then
    assert jnp.array_equal(flags, jnp.array([True, True, True])).item()


def test_marks_only_forward_neighbors_when_start_out_of_order():
    """Behavior: Flag positions only for valid forward transitions.

    Given:
      - A study list and a recall that begins out of order.
    When:
      - ``tabulate_trial`` processes the recalls.
    Then:
      - Only transitions that follow a +1 neighbor are marked.
    Why this matters:
      - Ensures scoring honors relative order.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3], dtype=jnp.int32)
    trial = jnp.array([2, 3, 1], dtype=jnp.int32)

    # Act / When
    flags = rspc.tabulate_trial(trial, presentation)

    # Assert / Then
    assert jnp.array_equal(flags, jnp.array([False, False, True])).item()


def test_ignores_zero_padding_when_recall_list_padded_with_zeros():
    """Behavior: Zero pads do not affect scoring.

    Given:
      - A study list and recalls padded with zeros.
    When:
      - ``tabulate_trial`` evaluates the trial.
    Then:
      - Only real recalls contribute to the flags.
    Why this matters:
      - Prevents padding from corrupting accuracy.
    """
    # Arrange / Given
    presentation = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    trial = jnp.array([1, 2, 3, 4, 0, 0], dtype=jnp.int32)

    # Act / When
    flags = rspc.tabulate_trial(trial, presentation, size=1)

    # Assert / Then
    assert jnp.array_equal(flags, jnp.array([True, True, True, True])).item()


def test_computes_mean_accuracy_when_multiple_trials():
    """Behavior: Average scores across trials.

    Given:
      - Recall and presentation matrices for two trials.
    When:
      - ``relative_spc`` is computed.
    Then:
      - Mean accuracy per position is returned.
    Why this matters:
      - Validates aggregation logic.
    """
    # Arrange / Given
    recalls = jnp.array([[1, 2, 3], [1, 2, 1]], dtype=jnp.int32)
    presentations = jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)

    # Act / When
    scores = rspc.relative_spc(recalls, presentations)

    # Assert / Then
    assert jnp.allclose(scores, jnp.array([1.0, 1.0, 0.5])).item()


def test_returns_axes_object_when_plotting_rspc():
    """Behavior: ``plot_relative_spc`` yields a Matplotlib ``Axes``.

    Given:
      - A minimal ``RecallDataset`` and trial mask.
    When:
      - ``plot_relative_spc`` is called.
    Then:
      - A Matplotlib ``Axes`` with a ``Figure`` is returned.
    Why this matters:
      - Confirms visualization utility uses standard objects.
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
    axis = rspc.plot_relative_spc(dataset, trial_mask)

    # Assert / Then
    assert isinstance(axis, Axes)
    fig = axis.figure
    assert isinstance(fig, Figure)
    plt.close(fig)
