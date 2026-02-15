import matplotlib

matplotlib.use("Agg", force=True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jaxcmr.analyses.termination_probability import (
    plot_termination_probability,
    simple_termination_probability,
    conditional_termination_probability,
)
from jaxcmr.helpers import make_dataset


def test_returns_point_mass_when_all_trials_stop_together():
    """Behavior: ``termination_probability`` concentrates mass at observed stop.

    Given:
      - Trials where recall halts immediately after the second output slot.
    When:
      - ``termination_probability`` runs in ``simple`` mode on the dataset.
    Then:
      - The termination probability vector equals [0.0, 1.0, 0.0, 0.0].
    Why this matters:
      - Confirms the analysis counts first zero per trial before plotting.
    """
    # Arrange / Given
    dataset = make_dataset(
        jnp.array(
            [
                [5, 0, 0, 0],
                [8, 0, 0, 0],
            ]
        )
    )

    # Act / When
    curve = simple_termination_probability(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(curve, expected).item()


def test_uses_final_slot_when_trials_never_enter_padding():
    """Behavior: ``termination_probability`` defaults to final slot when no zeros occur.

    Given:
      - Trials where one recall stream fills every slot while another stops after the first recall.
    When:
      - ``termination_probability`` runs in ``simple`` mode on the dataset.
    Then:
      - The termination probabilities split mass between the second and final slots.
    Why this matters:
      - Ensures the analysis remains well-defined for fully populated recall streams.
    """
    # Arrange / Given
    dataset = make_dataset(
        jnp.array(
            [
                [1, 2, 3, 4],
                [9, 0, 0, 0],
            ]
        )
    )

    # Act / When
    curve = simple_termination_probability(dataset)

    # Assert / Then
    expected = jnp.array([0.0, 0.5, 0.0, 0.5])
    assert jnp.allclose(curve, expected).item()


def test_returns_hazard_when_trials_drop_out():
    """Behavior: ``termination_probability`` divides by reach count in conditional mode.

    Given:
      - Trials where one stream ends at the third slot while another fills all slots.
    When:
      - ``termination_probability`` runs in ``conditional`` mode on the dataset.
    Then:
      - The conditional probability equals [0.0, 0.5, 0.0, 1.0].
    Why this matters:
      - Confirms the analysis reports the stop hazard rather than raw mass.
    """
    # Arrange / Given
    dataset = make_dataset(
        jnp.array(
            [
                [1, 2, 3, 4],
                [7, 0, 0, 0],
            ]
        )
    )

    # Act / When
    hazard = conditional_termination_probability(dataset)

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
    dataset = make_dataset(
        jnp.array(
            [
                [5, 0, 0],
                [4, 3, 0],
            ]
        )
    )
    trial_mask = jnp.array([True, True], dtype=bool)

    # Act / When
    axis_uncond = plot_termination_probability(dataset, trial_mask, mode="simple")
    axis_cond = plot_termination_probability(dataset, trial_mask, mode="conditional")

    # Assert / Then
    for axis in (axis_uncond, axis_cond):
        assert isinstance(axis, Axes)
        fig = axis.figure
        assert isinstance(fig, Figure)
        plt.close(fig)
