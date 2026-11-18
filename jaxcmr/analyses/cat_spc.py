"""Recall probability by study position within a category.

For each study position, ``cat_spc`` asks how often the item was recalled on
trials where an item of the specified category occupied that position.
Suppose a fruit appeared first on two lists and third on eight. The two
points on the fruit curve have denominators of two and eight respectively.
A position with no eligible fruit has an undefined probability.

The dataset supplies the category at each presentation. Recalling an item
credits each eligible presentation of it once. ``plot_cat_spc`` averages the
participant curves. To pool eligible presentations across positions into
one rate per category, use ``cat_recall_rate``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, Integer, RecallDataset
from ._study_positions import (
    last_study_position,
    recall_status,
    recalled_positions,
    study_position_mask,
)

__all__ = ["cat_spc", "category_recall_counts", "plot_cat_spc"]


def category_recall_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recall counts per position restricted to a category."""
    position_counts = recalled_positions(recalls[None, :], list_length)[0]
    return position_counts * (categories == category_value)


def fixed_pres_cat_spc(
    recalls: Integer[Array, " trial_count recall_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Category-filtered recall rate by study position.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_positions"]
        Recalled items (1-indexed; 0 for no recall).
    categories : Integer[Array, " trial_count study_positions"]
        Item categories per study position.
    category_value : int
        Category to filter on.
    list_length : int
        Study-list length.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall rate at each position for the category.

    """

    recall_counts = vmap(
        category_recall_counts,
        in_axes=(0, 0, None, None),
    )(recalls, categories, category_value, list_length)

    numerator = recall_counts.sum(axis=0)
    denominator = jnp.sum(categories == category_value, axis=0)
    return numerator / denominator


def cat_spc(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
) -> Float[Array, " study_positions"]:
    """Category-filtered recall rate by study position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item category metadata.
    category_field : str
        Key providing categories aligned with the ``pres_itemnos`` array.
    category_value : int
        Category to filter on.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall rate at each position for the category.

    """
    eligible = study_position_mask(dataset) & (
        dataset[category_field] == category_value
    )
    return (recall_status(dataset) & eligible).sum(axis=0) / eligible.sum(axis=0)


def plot_cat_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot category-filtered SPC curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    category_field : str
        Key providing categories aligned with the ``pres_itemnos`` array.
    category_values : Sequence[int]
        Categories to plot.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
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
        Axes with the category SPC plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        size = len(category_values) if len(category_values) > 1 else len(datasets)
        labels = [""] * size

    for data_index, data in enumerate(datasets):
        last_position = last_study_position(data, trial_masks[data_index])
        for label_index, category_value in enumerate(category_values):
            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(cat_spc, static_argnames=("category_field", "category_value")),
                    category_field=category_field,
                    category_value=category_value,
                )
            )

            color_idx = data_index * len(category_values) + label_index
            color = color_cycle[color_idx % len(color_cycle)]
            plot_mean_curve(
                axis,
                jnp.arange(last_position, dtype=int) + 1,
                subject_values[:, :last_position],
                labels[color_idx]
                if len(labels) > len(category_values)
                else (
                    labels[label_index]
                    if len(category_values) > 1
                    else labels[data_index]
                ),
                color,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )

    format_analysis_axes(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
