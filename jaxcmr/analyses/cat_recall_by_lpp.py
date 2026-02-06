"""Category-filtered LPP-binned recall analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from jax.nn import one_hot
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = [
    "category_recall_counts",
    "category_lpp_recall_histogram",
    "cat_recall_by_lpp",
    "plot_cat_recall_by_lpp",
]


def category_recall_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recall counts per position restricted to a category."""
    position_counts = jnp.bincount(recalls, length=list_length + 1)[1:]
    return position_counts * (categories == category_value)


def category_lpp_recall_histogram(
    recalls: Integer[Array, " trial_count recall_positions"],
    lpp: Float[Array, " trial_count study_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
    bin_edges: Float[Array, " bin_count_plus_one"],
    list_length: int,
) -> Float[Array, " bin_count"]:
    """Category-filtered recall rate binned by LPP.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_positions"]
        Recalled items (1-indexed; 0 for no recall).
    lpp : Float[Array, " trial_count study_positions"]
        LPP values per study position.
    categories : Integer[Array, " trial_count study_positions"]
        Item categories per study position.
    category_value : int
        Category to filter on.
    bin_edges : Float[Array, " bin_count_plus_one"]
        LPP bin edges.
    list_length : int
        Study-list length.

    Returns
    -------
    Float[Array, " bin_count"]
        Recall rate per LPP bin.

    """
    thresholds = bin_edges[1:-1]
    num_bins = bin_edges.shape[0] - 1
    bin_indices = jnp.digitize(lpp, thresholds)
    bin_one_hot = one_hot(bin_indices, num_bins, dtype=lpp.dtype)
    matches = (categories == category_value).astype(lpp.dtype)

    recall_counts = vmap(
        category_recall_counts,
        in_axes=(0, 0, None, None),
    )(recalls, categories, category_value, list_length)

    exposure_counts = (matches[..., None] * bin_one_hot).sum(axis=(0, 1))
    recall_bin_counts = (recall_counts[..., None] * bin_one_hot).sum(axis=(0, 1))
    exposure_counts = exposure_counts.astype(lpp.dtype)
    return recall_bin_counts / exposure_counts


def cat_recall_by_lpp(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    lpp_field: str = "LateLPP",
    bin_edges: Optional[Float[Array, " bin_count_plus_one"]] = None,
    bin_count: int = 10,
) -> Float[Array, " bin_count"]:
    """Category-filtered recall rate by LPP bins.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item LPP metadata.
    category_field : str
        Key providing item categories per study position.
    category_value : int
        Category to filter on.
    lpp_field : str
        Key providing LPP values per study position.
    bin_edges : Float[Array, " bin_count_plus_one"], optional
        LPP bin edges; computed from data if ``None``.
    bin_count : int
        Number of bins when ``bin_edges`` is not provided.

    Returns
    -------
    Float[Array, " bin_count"]
        Recall rate per LPP bin.

    """
    recalls = dataset["recalls"]
    lpp = dataset[lpp_field]
    categories = dataset[category_field]
    list_length = categories.shape[1]

    if bin_edges is None:
        lpp_min = jnp.min(lpp)
        lpp_max = jnp.max(lpp)
        bin_edges = jnp.linspace(lpp_min, lpp_max, bin_count + 1)

    return category_lpp_recall_histogram(
        recalls, lpp, categories, category_value, bin_edges, list_length
    )


def plot_cat_recall_by_lpp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    lpp_field: str = "LateLPP",
    bin_edges: Optional[Float[Array, " bin_count_plus_one"]] = None,
    bin_count: int = 10,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot category-filtered recall rate by LPP bins.

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
    lpp_field : str
        Key providing LPP values per study position.
    bin_edges : Float[Array, " bin_count_plus_one"], optional
        LPP bin edges; computed from data if ``None``.
    bin_count : int
        Number of bins when ``bin_edges`` is not provided.
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
        Axes with the LPP-binned recall plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        size = len(category_values) if len(category_values) > 1 else len(datasets)
        labels = [""] * size

    if bin_edges is None:
        minima = []
        maxima = []
        for data_index, data in enumerate(datasets):
            masked_lpp = data[lpp_field][trial_masks[data_index]]
            minima.append(jnp.min(masked_lpp))
            maxima.append(jnp.max(masked_lpp))
        global_min = jnp.min(jnp.stack(minima))
        global_max = jnp.max(jnp.stack(maxima))
        bin_edges = jnp.linspace(global_min, global_max, bin_count + 1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    color_index = 0
    for data_index, data in enumerate(datasets):
        for label_index, category_value in enumerate(category_values):
            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(
                        cat_recall_by_lpp,
                        static_argnames=(
                            "category_field",
                            "category_value",
                            "lpp_field",
                            "bin_count",
                        ),
                    ),
                    category_field=category_field,
                    category_value=category_value,
                    lpp_field=lpp_field,
                    bin_edges=bin_edges,
                    bin_count=bin_count,
                )
            )

            color = color_cycle[color_index % len(color_cycle)]
            color_index += 1
            plot_data(
                axis,
                bin_centers,
                subject_values,
                labels[label_index] if len(category_values) > 1 else labels[data_index],
                color,
                confidence_level=confidence_level,
            )

    set_plot_labels(axis, f"{lpp_field} (uV)", "Recall Rate", contrast_name)
    return axis
