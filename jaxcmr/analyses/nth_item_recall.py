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
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
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


def extra_conditional_nth_item_recall_curve(
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
    continuation = recalls != 0

    valid = jnp.logical_and(availability, continuation)
    numerator = (matches & valid).sum(axis=0)
    denominator = valid.sum(axis=0)
    return numerator / denominator


def conditional_nth_item_recall_curve(
    dataset: RecallDataset,
    query_study_position: int = 0,
) -> Float[Array, " recall_positions"]:
    """Returns nth-item recall rate by output position conditional on availability and a prior recall event.

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
    # Previous recall must be an actual item (True for the first slot by convention).
    previous_is_item = jnp.concatenate(
        (
            jnp.ones((recalls.shape[0], 1), dtype=bool),
            recalls[:, :-1] != 0,
        ),
        axis=1,
    )

    valid = jnp.logical_and(availability, previous_is_item)
    numerator = (matches & valid).sum(axis=0)
    denominator = valid.sum(axis=0)
    return numerator / denominator


def plot_conditional_nth_item_recall_curve(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_study_position: int = 0,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot the recall-position curve for a specific study position.

    Args:
      datasets: Collection of recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      query_study_position: Zero-based study position to analyze.
      color_cycle: Colors for successive datasets.
      labels: Legend labels for each dataset.
      contrast_name: Optional legend title.
      axis: Existing Matplotlib axis to draw on.
      confidence_level: Confidence level for the bounds.


    Returns:
      The Matplotlib axis containing the plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

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
            apply_by_subject(data, trial_masks[index], curve_fn, query_study_position)
        )

        subject_curves = subject_curves[:, :max_recall_length]
        recall_positions = jnp.arange(max_recall_length, dtype=jnp.int32) + 1

        color = color_cycle[index % len(color_cycle)]
        plot_data(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
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
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot the recall-position curve without availability gating.

    Args:
      datasets: Collection of recall datasets to plot.
      trial_masks: Boolean masks selecting trials in each dataset.
      query_study_position: Zero-based study position to analyze.
      color_cycle: Colors for successive datasets.
      labels: Legend labels for each dataset.
      contrast_name: Optional legend title.
      axis: Existing Matplotlib axis to draw on.
      confidence_level: Confidence level for the bounds.

    Returns:
      The Matplotlib axis containing the plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

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
            apply_by_subject(data, trial_masks[index], curve_fn, query_study_position)
        )

        subject_curves = subject_curves[:, :max_recall_length]
        recall_positions = jnp.arange(max_recall_length, dtype=jnp.int32) + 1

        color = color_cycle[index % len(color_cycle)]
        plot_data(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
        )

    item_number = query_study_position + 1
    set_plot_labels(
        axis,
        "Recall Position",
        f"P(Item {item_number})",
        contrast_name,
    )
    return axis
