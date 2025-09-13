"""Recall probability by lag.

Compute how recall varies with the spacing between repeated item presentations and
provide plotting utilities for visualizing repetition lag effects.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

__all__ = [
    "infer_max_lag",
    "item_lag_counts",
    "recall_probability_by_lag",
    "binned_recall_probability_by_lag",
    "plot_full_rpl",
    "plot_rpl",
]


def infer_max_lag(
    presentations: Integer[Array, " trials study_events"],
    list_length: int,
) -> int:
    """Return the largest first-to-second-presentation distance observed in ``presentations``."""
    all_items = jnp.arange(1, list_length + 1)
    positions = vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(None, 0, None)
    )(all_items, presentations, 2)
    return jnp.max(jnp.maximum(positions[..., 1] - positions[..., 0], 0)).item() - 1


def item_lag_counts(
    target_item: Int_,
    recalls: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    max_lag: int,
    n_bins: int,
) -> tuple[Bool[Array, " lag_bins"], Bool[Array, " lag_bins"]]:
    """Return one-hot vectors (presented, recalled) for ``target_item``'s lag bin.

    Args:
        target_item: Item ID to tabulate.
        recalls: First study positions of recalled items (1-indexed; 0 marks no recall).
        presentation: Item IDs presented at each study position (1-indexed).
        max_lag: Largest explicit lag bucket.
            - bin 0 → single presentation
            - bin k → k intervening items for 1 ≤ k ≤ ``max_lag``
            - bin ``max_lag + 1`` → lag exceeds ``max_lag``
        n_bins: Total number of bins (``max_lag + 2``).

    Each vector has at most one ``True``. Summing over items or trials yields counts.
    """
    first_pos, second_pos = item_to_study_positions(target_item, presentation, size=2)
    lag = jnp.maximum(second_pos - first_pos, 0)
    bin_index = jnp.clip(lag, 0, max_lag + 1)

    presented = (jnp.arange(n_bins) == bin_index) & (first_pos > 0)
    recalled = presented & jnp.any(recalls == first_pos)

    return presented, recalled


def _trial_lag_counts(
    recalls: Integer[Array, " recall_events"],
    presentations: Integer[Array, " study_events"],
    all_items: Integer[Array, " items"],
    max_lag: int,
    n_bins: int,
) -> tuple[Float[Array, " lag_bins"], Float[Array, " lag_bins"]]:
    """Aggregate presented and recalled one-hot vectors for one trial."""
    presented, recalled = vmap(item_lag_counts, in_axes=(0, None, None, None, None))(
        all_items, recalls, presentations, max_lag, n_bins
    )
    return presented.sum(0), recalled.sum(0)


def recall_probability_by_lag(
    recalls: Integer[Array, " trials recall_events"],
    presentations: Integer[Array, " trials study_events"],
    list_length: int,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Return recall probability for lag 0, 1, ..., ``max_lag``, plus once-presented items.

    Args:
        recalls: First study positions of recalled items (1-indexed; 0 marks no recall).
        presentations: Item IDs shown at each study position (1-indexed).
        list_length: Number of items in the study list.
        max_lag: Largest explicit lag bucket.
            - bin 0 → single presentation
            - bin k → k intervening items for 1 ≤ k ≤ ``max_lag``
            - bin ``max_lag + 1`` → lag exceeds ``max_lag``
    """
    n_bins = max_lag + 2
    all_items = jnp.arange(1, list_length + 1)

    presented_t, recalled_t = vmap(_trial_lag_counts, in_axes=(0, 0, None, None, None))(
        recalls, presentations, all_items, max_lag, n_bins
    )

    presented_tot = presented_t.sum(0)
    recalled_tot = recalled_t.sum(0)

    return jnp.where(presented_tot > 0, recalled_tot / presented_tot, 0.0)


def binned_recall_probability_by_lag(
    recalls: Integer[Array, " trials recall_events"],
    presentations: Integer[Array, " trials study_events"],
    list_length: int,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Return binned recall probability for lag 0, 1, ..., ``max_lag``, plus once-presented items.

    Args:
        recalls: First study positions of recalled items (1-indexed; 0 marks no recall).
        presentations: Item IDs shown at each study position (1-indexed).
        list_length: Number of items in the study list.
        max_lag: Largest explicit lag bucket.
            - bin 0 → single presentation
            - bin k → k intervening items for 1 ≤ k ≤ ``max_lag``
            - bin ``max_lag + 1`` → lag exceeds ``max_lag``
    """
    result = recall_probability_by_lag(recalls, presentations, list_length, max_lag)
    return (
        jnp.zeros(5)
        .at[0]
        .set(result[0])
        .at[1]
        .set(result[1])
        .at[2]
        .set((result[2] + result[3]) / 2)
        .at[3]
        .set((result[4] + result[5] + result[6]) / 3)
        .at[4]
        .set((result[7] + result[8] + result[9]) / 3)
    )


def plot_full_rpl(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """Return ``Axes`` with repetition-lag curves for each dataset and mask.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Boolean masks to filter trials.
        distances: Unused placeholder for interface compatibility.
        color_cycle: Colors for plotting each dataset.
        labels: Names for each dataset for the legend.
        contrast_name: Legend title for the plotted contrast.
        axis: Existing Matplotlib axes to plot on.
        size: Maximum number of study positions an item can be presented at.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    max_list_length = find_max_list_length(datasets, trial_masks)
    max_lag = infer_max_lag(datasets[0]["pres_itemnos"], max_list_length)
    xticklabels = ["N/A"] + [f"{i}" for i in range(max_lag + 1)]
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(
                    recall_probability_by_lag,
                    static_argnames=("max_lag", "list_length"),
                ),
                max_lag,
            )
        )

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(len(xticklabels)),
            subject_values,
            labels[data_index],
            color,
        )
    axis.set_xticklabels(xticklabels) # type: ignore
    set_plot_labels(axis, "Lag", "Recall Rate", contrast_name)
    return axis


def plot_rpl(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """Return ``Axes`` with binned repetition-lag curves for each dataset and mask.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Boolean masks to filter trials.
        distances: Unused placeholder for interface compatibility.
        color_cycle: Colors for plotting each dataset.
        labels: Names for each dataset for the legend.
        contrast_name: Legend title for the plotted contrast.
        axis: Existing Matplotlib axes to plot on.
        size: Maximum number of study positions an item can be presented at.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    max_list_length = find_max_list_length(datasets, trial_masks)
    max_lag = infer_max_lag(datasets[0]["pres_itemnos"], max_list_length)
    xticklabels = ["N/A", "0", "1-2", "3-5", "6-8"]
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(
                    binned_recall_probability_by_lag,
                    static_argnames=("list_length", "max_lag"),
                ),
                max_lag,
            )
        )

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(5),
            subject_values,
            labels[data_index],
            color,
        )

    axis.set_xticklabels(xticklabels) # type: ignore
    set_plot_labels(axis, "Lag", "Recall Rate", contrast_name)
    return axis
