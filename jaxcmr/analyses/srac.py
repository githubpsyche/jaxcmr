"""Compute and plot serial recall accuracy.

The Serial Recall Accuracy Curve (SRAC) reports the
proportion of trials on which the item studied at each
position is recalled in the matching output position.
"""

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["trial_srac", "srac", "plot_srac"]


def trial_srac(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Return a flag for each study position indicating correct recall.

    Args:
      recalls: Recalled item indices for one trial. Shape
        ``[recall_positions]``; 1-indexed with 0 for no recall.
      presentations: Item identifiers in study order. Shape
        ``[study_positions]``; 1-indexed.
      size: Maximum number of study positions an item can occupy.
    """
    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )
    study_positions = jnp.arange(1, len(recalls) + 1)
    return vmap(lambda r, i: jnp.any(r == i))(expanded_recalls, study_positions)


def srac(
    recalls: Integer[Array, " trial_count recall_positions"],
    presentations: Integer[Array, " trial_count study_positions"],
    list_length: int,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Return the proportion of correct recalls for each study position.

    Args:
      recalls: Trial by recall position array of recalled items. Shape
        ``[trial_count, recall_positions]``; 1-indexed with 0 for no recall.
      presentations: Trial by study position array of presented items. Shape
        ``[trial_count, study_positions]``; 1-indexed.
      list_length: Length of the study list. Included for API compatibility and
        ignored.
      size: Maximum number of study positions an item can occupy.
    """
    return vmap(trial_srac, in_axes=(0, 0, None))(
        recalls,
        presentations,
        size,
    ).mean(axis=0)


def plot_srac(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot serial recall accuracy curves for one or more datasets.

    Args:
      datasets: Recall datasets to plot.
      trial_masks: Boolean masks selecting trials for each dataset.
      color_cycle: Line colors for each dataset.
      labels: Legend entries corresponding to each dataset.
      contrast_name: Name of the contrast used in labeling.
      axis: Existing Matplotlib axes to draw on.

    Returns:
      The Matplotlib axes containing the plot.
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

    # Identify the largest list length across datasets so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(srac),
            )
        )

        color = color_cycle.pop(0)
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_data(axis, xvals, subject_values, labels[data_index], color)

    set_plot_labels(axis, "Study Position", "Serial Recall Accuracy", contrast_name)
    return axis
