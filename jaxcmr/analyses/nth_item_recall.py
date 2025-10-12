"""Compute nth-item recall curves.

Compute and visualize the probability that a studied position is produced
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

__all__ = [
    "simple_nth_item_recall_curve",
    "conditional_nth_item_recall_curve",
    "plot_simple_nth_item_recall_curve",
    "plot_conditional_nth_item_recall_curve",
]


def simple_nth_item_recall_curve(
    dataset: RecallDataset,
    query_study_position: int = 0,
) -> Float[Array, " recall_positions"]:
    """Returns recall-position probability of the queried studied item.

    Args:
      dataset: Recall dataset containing ``recalls`` and ``pres_itemnos``.
      query_study_position: Zero-based study position to analyze.
    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    queried_items = presentations[:, query_study_position]
    matches = recalls == queried_items[:, None]
    counts = matches.sum(axis=0)
    denominator = matches.shape[0]
    return counts / denominator


def conditional_nth_item_recall_curve(
    dataset: RecallDataset,
    query_study_position: int = 0,
) -> Float[Array, " recall_positions"]:
    """Returns nth-item recall rate by output position conditional on availability and continuation.

    Args:
      dataset: Recall dataset containing ``recalls`` and ``pres_itemnos``.
      query_study_position: Zero-based study position to analyze.
    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    queried_items = presentations[:, query_study_position]

    matches = recalls == queried_items[:, None]
    prior_recalls = jnp.cumsum(matches, axis=1) - matches
    availability = prior_recalls == 0
    
    valid = jnp.logical_and(availability, recalls != 0)
    numerator = (matches & valid).sum(axis=0)
    denominator = valid.sum(axis=0)
    return numerator/denominator


def plot_conditional_nth_item_recall_curve(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_study_position: int = 0,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot the recall-position curve for a specific study position.

    Args:
      datasets: Collection of recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      query_study_position: Zero-based study position to analyze.
      distances: Precomputed item distances retained for backward compatibility.
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

    curve_fn = conditional_nth_item_recall_curve

    for index, data in enumerate(datasets):
        subject_curves = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                curve_fn,
                query_study_position
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

    item_number = query_study_position + 1
    set_plot_labels(
        axis,
        "Recall Position",
        f"P(Item {item_number} | Available & Continuing)",
        contrast_name,
    )
    return axis


def plot_simple_nth_item_recall_curve(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_study_position: int = 0,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot the recall-position curve without availability gating.

    Args:
      datasets: Collection of recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      query_study_position: Zero-based study position to analyze.
      distances: Precomputed item distances retained for backward compatibility.
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

    curve_fn = jit(simple_nth_item_recall_curve)

    for index, data in enumerate(datasets):
        subject_curves = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                curve_fn,
                query_study_position
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

    item_number = query_study_position + 1
    set_plot_labels(
        axis,
        "Recall Position",
        f"P(Item {item_number})",
        contrast_name,
    )
    return axis
