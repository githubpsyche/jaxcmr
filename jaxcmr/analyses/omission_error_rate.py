"""Compute omission error rates for serial recall data."""

__all__ = ["trial_omission_error_rate", "omission_error_rate", "plot_omission_error_rate"]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset


def trial_omission_error_rate(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Returns omission flags for a single trial.

    Args:
      recalls: recall positions for a single trial. 1-indexed; 0 for no recall.
      presentations: study positions for a single trial. 1-indexed.
      size: number of studied items in the trial.
    """

    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )

    study_positions = jnp.arange(1, presentations.shape[0] + 1)

    position_was_recalled = vmap(lambda pos: jnp.any(expanded_recalls == pos))(
        study_positions
    )

    return (~position_was_recalled) & (presentations != 0)


def omission_error_rate(
    recalls: Integer[Array, " trial_count recall_positions"],
    presentations: Integer[Array, " trial_count study_positions"],
    list_length: int,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Returns position-specific omission rates.

    Args:
      recalls: trial by recall position array of recalled items. 1-indexed; 0 for no recall.
      presentations: trial by study position array of presented items. 1-indexed.
      list_length: unused; kept for API compatibility.
      size: number of studied items in each trial.
    """

    return vmap(trial_omission_error_rate, in_axes=(0, 0, None))(
        recalls, presentations, size
    ).mean(axis=0)

def plot_omission_error_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot omission rates for one or more datasets.

    Args:
      datasets: datasets containing trial data to be plotted.
      trial_masks: masks to filter trials in datasets.
      color_cycle: colors for each dataset.
      labels: labels for each dataset.
      contrast_name: label for the legend.
      axis: existing axes to plot on.
    """

    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [c["color"] for c in rcParams["axes.prop_cycle"]]

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    if labels is None:
        labels = ["" for _ in datasets]

    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict, trial_masks[data_index], jit(omission_error_rate)
            )
        )

        color = color_cycle.pop(0)
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_data(axis, xvals, subject_values, labels[data_index], color)

    set_plot_labels(axis, "Study Position", "Omission Error Rate", contrast_name)
    return axis
