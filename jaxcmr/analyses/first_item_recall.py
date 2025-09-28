"""First-item recall curves.

Compute and visualize the probability that the first studied item is produced
at each recall (output) position.
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

__all__ = ["first_item_recall_curve", "plot_first_item_recall_curve"]


def first_item_recall_curve(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Return the recall-position probability of the list's first item.

    Args:
      dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.

    Returns:
      A 1-D float array whose ``i``-th entry is the probability that the first
      studied item was produced at recall position ``i``.
    """

    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    first_items = presentations[:, 0]
    matches = recalls == first_items[:, None]
    counts = matches.sum(axis=0)
    denominator = matches.shape[0]
    return counts.astype(jnp.float32) / denominator


def plot_first_item_recall_curve(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot the recall-position curve for the first studied item.

    Args:
      datasets: Collection of recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      color_cycle: Colors for successive datasets.
      labels: Legend labels for each dataset.
      contrast_name: Optional legend title.
      axis: Existing Matplotlib axis to draw on.

    Returns:
      The Matplotlib axis containing the plot.
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

    curve_fn = jit(first_item_recall_curve)

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

    set_plot_labels(
        axis,
        "Recall Position",
        "P(Recall first studied item)",
        contrast_name,
    )
    return axis
