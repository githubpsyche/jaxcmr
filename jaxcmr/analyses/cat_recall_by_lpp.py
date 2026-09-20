"""LPP conveniences for the generic category-by-factor recall analysis.

LPP values are binned using ``cat_recall_by_factor``. Recall codes identify
one-indexed study positions; repeated responses count once per position.
Missing/nonfinite and out-of-range LPP values are excluded, and the final bin
includes its right edge. Empty bins return NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

from jax import numpy as jnp
from matplotlib.axes import Axes

from ..typing import Array, Bool, Float, Integer, RecallDataset
from .cat_recall_by_factor import (
    cat_recall_by_factor,
    category_factor_recall_histogram,
    plot_cat_recall_by_factor,
)

__all__ = [
    "cat_recall_by_lpp",
    "category_lpp_recall_histogram",
    "category_recall_counts",
    "plot_cat_recall_by_lpp",
]


def category_recall_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Return zero/one recalled-position counts restricted to a category.

    Retains the original public name; duplicate responses now count only once.
    """
    recalled = jnp.bincount(recalls, length=list_length + 1)[1:] > 0
    return (recalled & (categories == category_value)).astype(jnp.float32)


def category_lpp_recall_histogram(
    recalls: Integer[Array, " trial_count recall_positions"],
    lpp: Float[Array, " trial_count study_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
    bin_edges: Float[Array, " bin_count_plus_one"],
    list_length: int,
) -> Float[Array, " bin_count"]:
    """Return category-filtered recall rates in fixed LPP bins.

    ``list_length`` is retained for compatibility and must match metadata width.
    Use the dataset wrapper to exclude padded/invalid study positions, or the
    generic histogram's ``valid`` argument when working directly with arrays.
    """
    if lpp.shape[1] != list_length:
        raise ValueError("list_length must match the LPP study-position width.")
    return category_factor_recall_histogram(
        recalls, lpp, categories, category_value, bin_edges
    )


def cat_recall_by_lpp(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    lpp_field: str = "LateLPP",
    bin_edges: Float[Array, " bin_count_plus_one"] | None = None,
    bin_count: int = 10,
    *,
    valid_field: str | None = None,
) -> Float[Array, " bin_count"]:
    """Pool study-position recall rates by category and LPP bin.

    Delegates to ``cat_recall_by_factor``. Automatic edges use valid study
    values across categories; reuse explicit edges when comparing groups.
    ``valid_field`` optionally selects finite, positive quality/eligibility
    flags. Presentation padding and positions beyond listLength are excluded.
    """
    return cat_recall_by_factor(
        dataset, category_field, category_value, lpp_field, bin_edges, bin_count,
        valid_field=valid_field,
    )


def plot_cat_recall_by_lpp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    lpp_field: str = "LateLPP",
    bin_edges: Float[Array, " bin_count_plus_one"] | None = None,
    bin_count: int = 10,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
) -> Axes:
    """Plot subject-mean LPP-binned rates using shared reference bins.

    Retains the LPP x-axis label and original ``labels`` convention: labels name
    categories when more than one is requested, otherwise they name datasets.
    The generic plotter supports independent dataset and category labels.
    """
    return plot_cat_recall_by_factor(
        datasets, trial_masks, category_field, category_values, lpp_field,
        bin_edges=bin_edges,
        bin_count=bin_count,
        color_cycle=color_cycle,
        labels=labels if len(category_values) <= 1 else None,
        category_labels=labels if len(category_values) > 1 else None,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        valid_field=valid_field,
        factor_label=f"{lpp_field} (uV)",
    )
