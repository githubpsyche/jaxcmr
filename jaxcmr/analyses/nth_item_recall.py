"""Output positions at which a chosen study item is recalled.

When does the first item on a list come back? With ``query_study_position=0``,
each point on the curve is an output position for that first item. On a trial
with study list A, B, C and recalls C, A, the first item contributes at output
position 2. If the item was studied more than once, a recall recorded at any
of its study positions identifies the same item.

``simple_nth_item_recall_curve`` uses all trials containing the queried study
position as the denominator at every output. ``conditional_nth_item_recall_curve``
starts with those trials and, at each later output, counts only trials where
the item had yet to be recalled and the preceding output was nonzero. The two
plotters calculate their respective curves per participant before averaging.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, RecallDataset
from ._study_positions import study_position_mask

__all__ = [
    "conditional_nth_item_recall_curve",
    "plot_conditional_nth_item_recall_curve",
    "plot_simple_nth_item_recall_curve",
    "simple_nth_item_recall_curve",
]


def _recalls_of_study_item(
    dataset: RecallDataset, query_study_position: int
) -> tuple[Bool[Array, " trials recall_positions"], Bool[Array, " trials"]]:
    """Locate outputs naming the queried item and trials presenting it."""
    recalls = dataset["recalls"]
    eligible = study_position_mask(dataset)
    items = jnp.where(eligible, dataset["pres_itemnos"], 0)
    response_items = jnp.take_along_axis(
        items, jnp.maximum(recalls - 1, 0), axis=1, mode="fill", fill_value=0
    )
    queried = items[:, query_study_position]
    present = queried > 0
    matches = (recalls > 0) & present[:, None] & (response_items == queried[:, None])
    return matches, present


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
    matches, queried_present = _recalls_of_study_item(dataset, query_study_position)
    counts = matches.sum(axis=0)
    denominator = queried_present.sum()
    return counts / denominator


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
    matches, queried_present = _recalls_of_study_item(dataset, query_study_position)
    recalls = dataset["recalls"]
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

    valid = availability & previous_is_item & queried_present[:, None]
    numerator = (matches & valid).sum(axis=0)
    denominator = valid.sum(axis=0)
    return numerator / denominator


def plot_conditional_nth_item_recall_curve(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_study_position: int = 0,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
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

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

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
        plot_mean_curve(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    item_number = query_study_position + 1
    format_analysis_axes(
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
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
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

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

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
        plot_mean_curve(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    item_number = query_study_position + 1
    format_analysis_axes(
        axis,
        "Recall Position",
        f"P(Item {item_number})",
        contrast_name,
    )
    return axis
