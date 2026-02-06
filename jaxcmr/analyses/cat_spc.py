"""Category-filtered serial position curve analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["category_recall_counts", "cat_spc", "plot_cat_spc"]


def category_recall_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recall counts per position restricted to a category."""
    position_counts = jnp.bincount(recalls, length=list_length + 1)[1:]
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
        Key providing item categories per study position.
    category_value : int
        Category to filter on.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall rate at each position for the category.

    """
    recalls = dataset["recalls"]
    categories = dataset[category_field]
    list_length = categories.shape[1]

    return fixed_pres_cat_spc(recalls, categories, category_value, list_length)


def plot_cat_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot category-filtered SPC curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    category_field : str
        Key providing item categories per study position.
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

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):
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
            plot_data(
                axis,
                jnp.arange(max_list_length, dtype=int) + 1,
                subject_values,
                labels[label_index] if len(category_values) > 1 else labels[data_index],
                color,
            )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
