"""Conditional co-recall probability by category relation.

Given that an item was recalled, computes the probability that a
same-category or different-category item at each study lag was also
recalled. Separates co-recall curves by within-category vs
between-category pairs.

"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["conditional_corec_by_cat", "plot_conditional_corec_by_cat"]


def _trial_recall_mask(
    recalls: Integer[Array, " recall_events"],
    list_length: int,
) -> Bool[Array, " study_positions"]:
    """Return a boolean mask of recalled study positions.

    Parameters
    ----------
    recalls : Integer[Array, " recall_events"]
        One-indexed recall positions for a single
        trial (0 pads).
    list_length : int
        Number of studied items in the list.

    """

    counts = jnp.bincount(recalls, length=list_length + 1)[1:]
    return counts > 0


def _conditional_category_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
) -> tuple[Float[Array, " relation"], Float[Array, " relation"]]:
    """Return category-conditioned co-recall counts for a trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_events"]
        One-indexed recall positions for a single
        trial (0 pads).
    categories : Integer[Array, " study_positions"]
        Category identifiers ordered by study
        position.
    category_value : int
        Reference category defining which recalled
        items act as anchors.

    Returns
    -------
    tuple[Float[Array, " relation"], Float[Array, " relation"]]
        Actual and possible same- and different-
        category co-recall counts.

    """

    mask = _trial_recall_mask(recalls, categories.shape[0]).astype(jnp.float32)
    pair_mask = jnp.ones((categories.shape[0], categories.shape[0]), dtype=mask.dtype)
    pair_mask = pair_mask - jnp.eye(categories.shape[0], dtype=mask.dtype)

    anchor_selector = (categories == category_value).astype(mask.dtype)
    anchor_mask = anchor_selector * mask
    same_indicator = anchor_selector[None, :]
    diff_indicator = 1.0 - same_indicator

    anchor_matrix = anchor_mask[:, None] * pair_mask
    neighbor_recalled = mask[None, :]

    actual_same = (anchor_matrix * same_indicator * neighbor_recalled).sum()
    actual_diff = (anchor_matrix * diff_indicator * neighbor_recalled).sum()
    possible_same = (anchor_matrix * same_indicator).sum()
    possible_diff = (anchor_matrix * diff_indicator).sum()

    actual = jnp.stack((actual_same, actual_diff))
    possible = jnp.stack((possible_same, possible_diff))
    return actual, possible


def conditional_corec_by_cat(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
) -> Float[Array, " relation"]:
    """Return conditional category-wise co-recall probabilities.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and
        the requested category field.
    category_field : str
        Key mapping to per-trial study-item
        categories.
    category_value : int
        Reference category defining the anchor items.

    Returns
    -------
    Float[Array, " relation"]
        Same- and different-category conditional
        co-recall probabilities.

    """

    categories = dataset[category_field]
    actual, anchors = vmap(_conditional_category_counts, in_axes=(0, 0, None))(
        dataset["recalls"], categories, category_value
    )
    return actual.sum(0) / anchors.sum(0)


def plot_conditional_corec_by_cat(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_value: int,
    relation_labels: Sequence[str] = ("Same", "Different"),
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot conditional co-recall by category with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset or list of datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials within each
        dataset.
    category_field : str
        Dataset key containing per-study-item
        categories.
    category_value : int
        Reference category defining the anchors.
    relation_labels : Sequence[str], optional
        Tick labels for same- and different-category
        neighbor probabilities.
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing Matplotlib ``Axes`` to plot on.
    confidence_level : float, optional
        Confidence level for the bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with conditional co-recall
        by category curves.

    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    relation_axis = jnp.arange(len(relation_labels), dtype=int)

    for index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                jit(
                    conditional_corec_by_cat,
                    static_argnames=("category_field", "category_value"),
                ),
                category_field=category_field,
                category_value=category_value,
            )
        )

        color = color_cycle[index % len(color_cycle)]
        plot_data(
            axis,
            relation_axis,
            subject_values,
            labels[index],
            color,
            confidence_level=confidence_level,
        )

    axis.set_xticks(relation_axis)
    axis.set_xticklabels(list(relation_labels))

    set_plot_labels(
        axis,
        "Category Relation",
        "Conditional Co-Recall Probability",
        contrast_name,
    )
    return axis
