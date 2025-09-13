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
    expected = jnp.array([False, False, False])
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

def test_srac_all_correct():
    presentations = jnp.array([[1, 2, 3]])
    recalls = jnp.array(
        [
            [1, 2, 3]  # correct: recalls[0] → presentations[0], etc.
        ]
    )
    expected = jnp.array([1.0, 1.0, 1.0])
    result = srac.trial_srac(recalls[0], presentations[0])
    assert jnp.allclose(result, expected), result


def test_srac_with_repetition():
    presentations = jnp.array([[1, 2, 1]])
    recalls = jnp.array(
        [
            [1, 2, 1]  # recall presentations[0], presentations[1], presentations[0]
        ]
    )
    # study pos 0: item 1, recalled item 1 → correct
    # study pos 1: item 2, recalled item 2 → correct
    # study pos 2: item 1, recalled item 1 (again) → correct
    expected = jnp.array([1.0, 1.0, 1.0])
    result = srac.srac(recalls, presentations, list_length=3)
    assert jnp.allclose(result, expected), result


def test_srac_with_no_recalls():
    presentations = jnp.array([[1, 2, 3]])
    recalls = jnp.array(
        [
            [0, 0, 0]  # no recalls
        ]
    )
    expected = jnp.array([0.0, 0.0, 0.0])
    result = srac.srac(recalls, presentations, list_length=3)
    assert jnp.allclose(result, expected), result


def test_srac_with_some_missing_and_errors():
    presentations = jnp.array([[1, 1, 2, 2]])
    recalls = jnp.array([[1, 1, 3, 0]])
    expected = jnp.array([1.0, 1.0, 1.0, 0])
    result = srac.srac(recalls, presentations, list_length=4)
    assert jnp.allclose(result, expected), result


def test_srac_deflates_accuracy_by_including_padded_positions():
    presentations = jnp.array(
        [
            [1, 2, 0],  # Trial 0: list length 2
            # [1, 2, 3],  # Trial 1: list length 3
        ]
    )
    recalls = jnp.array(
        [
            [1, 2, 0],  # Trial 0: correct
            # [1, 2, 3],  # Trial 1: correct
        ]
    )
    list_lengths = [2]

    # At position 2:
    # - Trial 0: padded, should NOT contribute
    # - Trial 1: correct recall → should yield 1.0

    # If code uses mean over all trials, it computes (0 + 1)/2 = 0.5 ← WRONG
    # Correct behavior: (1)/1 = 1.0
    expected = jnp.array([1.0, 1.0, 0.0])  # strict SRAC: don't penalize for padding

    result = srac.srac(presentations, recalls, list_lengths)
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"