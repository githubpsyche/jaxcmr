"""Compute and plot repetition-index serial recall accuracy.

The repetition-index SRAC reports accuracy at repeated study positions,
stratified by repetition index (first vs second occurrence, etc.).
"""

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["trial_repsrac_counts", "repsrac", "plot_repsrac"]


def trial_repsrac_counts(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> tuple[Integer[Array, " repetition_index"], Integer[Array, " repetition_index"]]:
    """Return correct and total counts for each repetition index in a trial.

    Args:
      recalls: Recalled item indices for one trial. Shape
        ``[recall_positions]``; 1-indexed with 0 for no recall.
      presentations: Item identifiers in study order. Shape
        ``[study_positions]``; 1-indexed.
      size: Maximum number of study positions an item can occupy.

    Returns:
      (correct_counts, total_counts): Counts per repetition index for repeated items.
    """
    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )
    study_positions = jnp.arange(1, presentations.shape[0] + 1)
    correct_positions = vmap(lambda r, i: jnp.any(r == i))(
        expanded_recalls, study_positions
    )
    item_positions = vmap(all_study_positions, in_axes=(0, None, None))(
        study_positions, presentations, size
    )
    repeated_mask = (
        item_positions[:, 1] > 0
        if size > 1
        else jnp.zeros_like(study_positions, dtype=bool)
    )
    matches = item_positions == study_positions[:, None]
    index_mask = matches & repeated_mask[:, None]
    correct_counts = jnp.sum(index_mask & correct_positions[:, None], axis=0)
    total_counts = jnp.sum(index_mask, axis=0)
    return correct_counts, total_counts


def repsrac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " repetition_index"]:
    """Return recall accuracy for each repetition index.

    Args:
      dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.
      size: Maximum number of study positions an item can occupy.
    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    correct_counts, total_counts = vmap(trial_repsrac_counts, in_axes=(0, 0, None))(
        recalls, presentations, size
    )
    total_correct = correct_counts.sum(axis=0)
    total_possible = total_counts.sum(axis=0)
    return jnp.where(total_possible > 0, total_correct / total_possible, 0.0)


def plot_repsrac(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """Plot repetition-index serial recall accuracy for one or more datasets.

    Args:
      datasets: Recall datasets to plot.
      trial_masks: Boolean masks selecting trials for each dataset.
      color_cycle: Line colors for each dataset.
      labels: Legend entries corresponding to each dataset.
      contrast_name: Name of the contrast used in labeling.
      axis: Existing Matplotlib axes to draw on.
      size: Maximum number of study positions an item may occupy.

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

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(repsrac, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle.pop(0)
        xvals = jnp.arange(size) + 1
        plot_data(axis, xvals, subject_values, labels[data_index], color)

    set_plot_labels(axis, "Repetition Index", "Serial Recall Accuracy", contrast_name)
    return axis
