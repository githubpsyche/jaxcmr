"""Compute termination probability curves.

These utilities summarize how often participants stop recalling at each
output position and provide plotting helpers for group comparisons.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, RecallDataset

__all__ = ["simple_termination_probability", "conditional_termination_probability"]


def simple_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Returns termination probability by recall position.

    Args:
      dataset: Recall dataset containing ``recalls``.
    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    counts = jnp.bincount(termination_index, length=recalls.shape[1])
    return counts / recalls.shape[0]


def conditional_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Returns conditional termination probability by recall position.

    Args:
      dataset: Recall dataset containing ``recalls``.
    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    stop_counts = jnp.bincount(termination_index, length=recalls.shape[1])
    recall_positions = jnp.arange(recalls.shape[1])
    reached_mask = recall_positions <= termination_index[:, None]
    reached_counts = jnp.sum(reached_mask, axis=0)
    return stop_counts / reached_counts

def plot_termination_probability(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    mode: str = "conditional",
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plots termination probability curves for the requested mode.

    Args:
      datasets: Recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      mode: `"conditional"` divides by reach counts; `"simple"` uses raw stops.
      distances: Reserved for API compatibility.
      color_cycle: Colors for successive datasets.
      labels: Legend labels for each dataset.
      contrast_name: Optional legend title.
      axis: Existing Matplotlib axis to draw on.

    Returns:
      Matplotlib axis with the rendered curves.
    """
    axis = init_plot(axis)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    if color_cycle is None:
        color_cycle = [entry["color"] for entry in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    max_recall_length = max(
        int(dataset["recalls"].shape[1])
        for dataset, mask in zip(datasets, trial_masks)
        if jnp.any(mask)
    )

    if mode == "conditional":
        curve_fn = jit(conditional_termination_probability)
    elif mode == "simple":
        curve_fn = jit(simple_termination_probability)

    for index, data in enumerate(datasets):
        subject_curves = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                curve_fn,
            )
        )

        subject_curves = subject_curves[:, :max_recall_length]
        recall_positions = jnp.arange(max_recall_length, dtype=jnp.int32) + 1

        color = color_cycle.pop(0)
        plot_data(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
        )

    ylabel = "P(Terminate | Reach)" if mode == "conditional" else "P(Terminate)"
    set_plot_labels(axis, "Recall Position", ylabel, contrast_name)
    return axis
