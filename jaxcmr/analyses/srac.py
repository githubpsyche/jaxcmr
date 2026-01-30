"""Compute and plot serial recall accuracy.

The Serial Recall Accuracy Curve (SRAC) reports the
proportion of trials on which the item studied at each
position is recalled in the matching output position.
"""

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
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
    list_length = presentations.shape[0]
    recalls = recalls[:list_length]
    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )
    study_positions = jnp.arange(1, list_length + 1)
    return vmap(lambda r, i: jnp.any(r == i))(expanded_recalls, study_positions)


def srac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Return the proportion of correct recalls for each study position.

    Args:
      dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.
      size: Maximum number of study positions an item can occupy.
    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

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
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot serial recall accuracy curves for one or more datasets.

    Args:
      datasets: Recall datasets to plot.
      trial_masks: Boolean masks selecting trials for each dataset.
      color_cycle: Line colors for each dataset.
      labels: Legend entries corresponding to each dataset.
      contrast_name: Name of the contrast used in labeling.
      axis: Existing Matplotlib axes to draw on.
      size: Maximum number of study positions an item may occupy.
      confidence_level: Confidence level for the bounds.

    Returns:
      The Matplotlib axes containing the plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(srac, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        xvals = jnp.arange(max_list_length) + 1
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_data(
            axis,
            xvals,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Serial Recall Accuracy", contrast_name)
    return axis
