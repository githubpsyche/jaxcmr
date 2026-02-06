"""Nth-item recall curves."""

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
    """Recall-position probability of the queried study item.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    query_study_position : int
        0-based study position to analyze.

    Returns
    -------
    Float[Array, " recall_positions"]
        Probability at each recall output position.

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
    """Nth-item recall conditioned on availability and continuation.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    query_study_position : int
        0-based study position to analyze.

    Returns
    -------
    Float[Array, " recall_positions"]
        Conditional probability at each output position.

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
    """Nth-item recall conditioned on availability and prior recall.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    query_study_position : int
        0-based study position to analyze.

    Returns
    -------
    Float[Array, " recall_positions"]
        Conditional probability at each output position.

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
    """Plot conditional nth-item recall curve.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    query_study_position : int
        0-based study position to analyze.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the recall curve.

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
    """Plot simple nth-item recall curve.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    query_study_position : int
        0-based study position to analyze.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the recall curve.

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
