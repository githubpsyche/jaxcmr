import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses.termination_position import (
    conditional_termination_position_curve,
    plot_conditional_termination_position_curve,
    plot_termination_position_curve,
    termination_position_curve,
)
from jaxcmr.typing import RecallDataset


def _make_dataset(recalls: jnp.ndarray) -> RecallDataset:
    recalls_arr = jnp.asarray(recalls, dtype=jnp.int32)
    trial_count, recall_slots = recalls_arr.shape
    return {
        "subject": jnp.ones((trial_count, 1), dtype=jnp.int32),
        "listLength": jnp.full((trial_count, 1), recall_slots, dtype=jnp.int32),
        "recalls": recalls_arr,
    }  # type: ignore


def test_returns_point_mass_when_all_trials_stop_together():
    """Behavior: ``termination_position_curve`` concentrates mass at observed stop.

    Given:
      - Trials where recall halts immediately after the second output slot.
    When:
      - ``termination_position_curve`` runs on the dataset.
    Then:
      - The termination probability vector equals [0.0, 1.0, 0.0, 0.0].
    Why this matters:
      - Confirms the analysis counts first zero per trial before plotting.
    """
    # Arrange / Given
    dataset = _make_dataset(
        jnp.array(
            [
                [5, 0, 0, 0],
                [8, 0, 0, 0],
            ]
        )
    )

    # Act / When
    curve = termination_position_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_uses_final_slot_when_trials_never_enter_padding():
    """Behavior: ``termination_position_curve`` defaults to final slot when no zeros occur.

    Given:
      - Trials where one recall stream fills every slot while another stops after the first recall.
    When:
      - ``termination_position_curve`` runs on the dataset.
    Then:
      - The termination probabilities split mass between the second and final slots.
    Why this matters:
      - Ensures the analysis remains well-defined for fully populated recall streams.
    """
    # Arrange / Given
    dataset = _make_dataset(
        jnp.array(
            [
                [1, 2, 3, 4],
                [9, 0, 0, 0],
            ]
        )
    )

    # Act / When
    curve = termination_position_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.5, 0.0, 0.5])
    assert jnp.allclose(curve, expected).item()


def test_returns_hazard_when_trials_drop_out():
    """Behavior: ``conditional_termination_position_curve`` divides by reach count.

    Given:
      - Trials where one stream ends at the third slot while another fills all slots.
    When:
      - ``conditional_termination_position_curve`` runs on the dataset.
    Then:
      - The conditional probability equals [0.0, 0.5, 0.0, 1.0].
    Why this matters:
      - Confirms the analysis reports the stop hazard rather than raw mass.
    """
    # Arrange / Given
    dataset = _make_dataset(
        jnp.array(
            [
                [1, 2, 3, 4],
                [7, 0, 0, 0],
            ]
        )
    )

    # Act / When
    hazard = conditional_termination_position_curve(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.5, 0.0, 1.0])
    assert jnp.allclose(hazard, expected).item()


def test_plot_functions_return_axes():
    """Behavior: Termination plots return Matplotlib ``Axes`` objects.

    Given:
      - A minimal ``RecallDataset`` with two trials.
    When:
      - The unconditional and conditional plotters are invoked.
    Then:
      - Each call yields an ``Axes`` associated with a ``Figure``.
    Why this matters:
      - Ensures the visualization helpers integrate with plotting workflows.
    """
    # Arrange / Given
    dataset = _make_dataset(
        jnp.array(
            [
                [5, 0, 0],
                [4, 3, 0],
            ]
        )
    )
    trial_mask = jnp.array([True, True], dtype=bool)

    # Act / When
    axis_uncond = plot_termination_position_curve(dataset, trial_mask)
    axis_cond = plot_conditional_termination_position_curve(dataset, trial_mask)

    # Assert / Then
    for axis in (axis_uncond, axis_cond):
        assert isinstance(axis, Axes)
        fig = axis.figure
        assert isinstance(fig, Figure)
        plt.close(fig)
