"""Probability of nth recall (PNR) metrics and plots.

Utilities for computing how often study positions produce the n-th recall,
with support for repeated study items.
"""

__all__ = [
    "fixed_pres_pnr",
    "available_recalls",
    "actual_recalls",
    "conditional_fixed_pres_pnr",
    "pnr",
    "available_recalls_with_repeats",
    "actual_recalls_with_repeats",
    "conditional_pnr_with_repeats",
    "plot_pnr",
]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap, lax
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..helpers import apply_by_subject, find_max_list_length
from ..typing import Array, Bool, Float, Integer, RecallDataset


def fixed_pres_pnr(
    recalls: Integer[Array, " trial_count recall_positions"],
    list_length: int,
    query_recall_position: int = 0,
) -> Float[Array, " study_positions"]:
    """Returns probability of n-th recall as a function of study position.

    Args:
        recalls: Trial by recall position array of recalled items. 1-indexed; 0 for no recall.
        list_length: Length of the study list.
        query_recall_position: Recall index (0-based) to analyze.
    """
    # Identify the item recalled at the query_recall_position for each trial.
    # Bin counts for each item number, ignoring 0 (no recall).
    # Divide by the total number of trials to get a probability.
    return jnp.bincount(
        recalls[:, query_recall_position].flatten(), length=list_length + 1
    )[1:] / len(recalls)


def available_recalls(
    recalls: Integer[Array, " recall_positions"],
    query_recall_position: int,
    list_length: int,
) -> Bool[Array, " list_length"]:
    """Returns mask of study positions still available at a recall position.

    Assumes recalls are 1-indexed with 0 meaning no recall.

    Args:
        recalls: Recalled items for a single trial.
        query_recall_position: Index in the recall sequence to evaluate.
        list_length: Length of the study list.
    """
    prior = recalls[:query_recall_position]
    init = jnp.ones(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, i: (m.at[i].set(False), None), init, prior)

    return final_mask[1:]


def actual_recalls(
    recalls: Integer[Array, " recall_positions"],
    query_recall_position: int,
    list_length: int,
) -> Bool[Array, " list_length"]:
    """Returns mask with the recalled study position set to True.

    Assumes items are 1-indexed with 0 for no recall.

    Args:
        recalls: Recalled items for a single trial.
        query_recall_position: Index in the recall sequence to evaluate.
        list_length: Length of the study list.
    """
    item = recalls[query_recall_position]
    return jnp.where(
        item == 0,
        jnp.zeros(list_length, dtype=bool),
        jnp.arange(1, list_length + 1) == item,
    )


def conditional_fixed_pres_pnr(
    recalls: Integer[Array, " trial recall_positions"],
    list_length: int,
    query_recall_position: int,
) -> Float[Array, " list_length"]:
    """Returns conditional PNR as actual over available per study position.

    Each element gives the probability that a study position was recalled
    at ``query_recall_position`` conditioned on it being available.

    Args:
        recalls: Trial by recall position array of recalled items. 1-indexed.
        list_length: Length of the study list.
        query_recall_position: Recall index (0-based) to analyze.
    """

    # shape (trial, list_length)
    actual = vmap(actual_recalls, in_axes=(0, None, None))(
        recalls, query_recall_position, list_length
    )
    available = vmap(available_recalls, in_axes=(0, None, None))(
        recalls, query_recall_position, list_length
    )

    numerator = actual.sum(axis=0)  # times each pos was the nth recall
    denominator = available.sum(axis=0)  # times each pos was available

    # Avoid divide-by-zero
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


def pnr(
    recalls: Integer[Array, " trial_count recall_positions"],
    presentations: Integer[Array, " trial_count study_positions"],
    list_length: int,
    size: int = 3,
    query_recall_position: int = 0,
) -> Float[Array, " study_positions"]:
    """Returns probability of n-th recall allowing item repetitions.

    Args:
        recalls: Trial by recall position array of recalled items. 1-indexed; 0 for no recall.
        presentations: Trial by study position array of presented items. 1-indexed.
        list_length: Length of the study list.
        size: Maximum number of study positions an item can be presented at.
        query_recall_position: Recall index (0-based) to analyze.
    """
    # expanded_recalls: (trial_count, recall_positions, size) array
    # where each recalled item is mapped to all its possible study positions.
    expanded_recalls = vmap(
        vmap(all_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, presentations, size)
    return fixed_pres_pnr(expanded_recalls, list_length, query_recall_position)


def available_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Returns mask of available study positions when items may repeat.

    A study position is unavailable if its item has been recalled earlier.

    Args:
        recalls: Recalled items for a single trial.
        presentations: Items presented at each study position.
        query_recall_position: Recall index to evaluate.
        list_length: Length of the study list.
        size: Maximum number of study positions an item can occupy.
    """
    prior = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls[:query_recall_position], presentations, size
    ).reshape(-1)

    init = jnp.ones(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(False), None), init, prior)

    return final_mask[1:]


def actual_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Returns mask with study positions of the recalled item set to True.

    Uses ``all_study_positions`` to handle item repetitions.

    Args:
        recalls: Recalled items for a single trial.
        presentations: Items presented at each study position.
        query_recall_position: Recall index to evaluate.
        list_length: Length of the study list.
        size: Maximum number of study positions an item can occupy.
    """
    item = recalls[query_recall_position]
    current = all_study_positions(item, presentations, size)  # shape: (size,)

    init = jnp.zeros(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(True), None), init, current)
    return final_mask[1:]


def conditional_pnr_with_repeats(
    recalls: Integer[Array, " trial recall_positions"],
    presentations: Integer[Array, " trial list_length"],
    list_length: int,
    size: int,
    query_recall_position: int,
) -> Float[Array, " list_length"]:
    """Returns conditional PNR when study items may repeat.

    Computes ``actual / available`` for each study position across trials.

    Args:
        recalls: Trial by recall position array of recalled items.
        presentations: Trial by study position array of presented items.
        list_length: Length of the study list.
        size: Maximum number of study positions an item can occupy.
        query_recall_position: Recall index (0-based) to analyze.
    """
    actual = vmap(actual_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )
    available = vmap(available_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )

    numerator = actual.sum(axis=0)
    denominator = available.sum(axis=0)

    return jnp.where(denominator > 0, numerator / denominator, 0.0)

def plot_pnr(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_recall_position: int = 0,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """Returns Axes object with plotted probability of nth recall for given datasets and trial masks.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        query_recall_position: Which recall index (0-based) to plot (e.g., 0 for first recall).
        distances: Unused, included for compatibility with other plotting functions.
        color_cycle: List of colors for plotting each dataset.
        labels: Names for each dataset for legend, optional.
        contrast_name: Name of contrast for legend labeling, optional.
        axis: Existing matplotlib Axes to plot on, optional.
        size: Maximum number of study positions an item can be presented at.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    # Convert single dict or single mask to list
    if isinstance(datasets, dict):
        datasets = [datasets]
    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    # Determine x-axis length
    max_list_length = find_max_list_length(datasets, trial_masks)

    # For each dataset, apply the pnr function by subject, gather results, and plot
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(conditional_pnr_with_repeats, static_argnames=("size", "list_length", "query_recall_position")),
                size,
                query_recall_position,
            )
        )

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(max_list_length, dtype=int) + 1,
            subject_values,
            labels[data_index],
            color,
        )

    set_plot_labels(axis, "Study Position", "Probability of Nth Recall", contrast_name)
    return axis
