"""Study positions of the items recalled at a chosen output position.

Where on the study list did the second recall come from? Set
``query_recall_position=1`` to get one value per study position for the second
recall. On a trial with study list A, B, C and recalls C, A, the second recall
credits study position 1. If A appeared more than once, it credits each of its
study positions.

``pnr`` divides the count at each study position by the total number of trials,
including trials with no second recall. ``conditional_pnr_with_repeats`` uses a
separate denominator for each position: the number of trials on which its item
had yet to be recalled before the chosen output. An item can still count as
available on a trial that ended early. ``plot_pnr`` averages these conditional
curves across participants.
"""

__all__ = [
    "actual_recalls",
    "actual_recalls_with_repeats",
    "available_recalls",
    "available_recalls_with_repeats",
    "conditional_fixed_pres_pnr",
    "conditional_pnr_with_repeats",
    "fixed_pres_pnr",
    "plot_pnr",
    "pnr",
    "recall_position_label",
]

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, lax, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset
from ._study_positions import last_study_position


def recall_position_label(query_recall_position: int) -> str:
    """Return a human-readable label for a 0-based recall position."""
    recall_number = query_recall_position + 1
    special = {1: "First", 2: "Second", 3: "Third"}
    if recall_number in special:
        return f"Probability of {special[recall_number]} Recall"
    if 10 <= recall_number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(recall_number % 10, "th")
    return f"Probability of {recall_number}{suffix} Recall"


def fixed_pres_pnr(
    recalls: Integer[Array, " trial_count recall_positions"],
    list_length: int,
    query_recall_position: int = 0,
) -> Float[Array, " study_positions"]:
    """Probability of nth recall by study position.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_positions"]
        1-indexed recall array; 0 for no recall.
    list_length : int
        Number of items in the study list.
    query_recall_position : int
        0-based recall index to analyze.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall probability at each study position.

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
    """Mask of study positions available at a recall position.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.

    Returns
    -------
    Bool[Array, " list_length"]
        True for positions not yet recalled.

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
    """Mask with the recalled study position set to True.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.

    Returns
    -------
    Bool[Array, " list_length"]
        True at the recalled position, False elsewhere.

    """
    item = recalls[query_recall_position]
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(list_length, dtype=bool),
        lambda: jnp.arange(1, list_length + 1) == item,
    )


def conditional_fixed_pres_pnr(
    recalls: Integer[Array, " trial recall_positions"],
    list_length: int,
    query_recall_position: int,
) -> Float[Array, " list_length"]:
    """Conditional PNR: actual over available per position.

    Parameters
    ----------
    recalls : Integer[Array, " trial recall_positions"]
        1-indexed recall array.
    list_length : int
        Number of items in the study list.
    query_recall_position : int
        0-based recall index to analyze.

    Returns
    -------
    Float[Array, " list_length"]
        Conditional probability at each study position.

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
    return numerator / denominator


def pnr(
    dataset: RecallDataset,
    size: int = 3,
    query_recall_position: int = 0,
) -> Float[Array, " study_positions"]:
    """Probability of nth recall with item repetitions.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.
    query_recall_position : int
        0-based recall index to analyze.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall probability at each study position.

    """
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]

    # expanded_recalls: (trial_count, recall_positions, size) array where each
    # recalled item is mapped to all its possible study positions.
    expanded_recalls = vmap(
        vmap(equivalent_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(dataset["recalls"], presentations, size)
    return fixed_pres_pnr(expanded_recalls, list_length, query_recall_position)


def available_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Mask of available positions when items may repeat.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    presentations : Integer[Array, " list_length"]
        Items presented at each study position.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " list_length"]
        True for positions not yet recalled.

    """
    prior = vmap(equivalent_study_positions, in_axes=(0, None, None))(
        recalls[:query_recall_position], presentations, size
    ).reshape(-1)

    init = jnp.ones(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(False), None), init, prior)

    return final_mask[1:] & (presentations > 0)


def actual_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Mask with study positions of the recalled item as True.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    presentations : Integer[Array, " list_length"]
        Items presented at each study position.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " list_length"]
        True at all positions of the recalled item.

    """
    item = recalls[query_recall_position]
    current = equivalent_study_positions(item, presentations, size)  # shape: (size,)

    init = jnp.zeros(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(True), None), init, current)
    return final_mask[1:]


def conditional_pnr_with_repeats(
    dataset: RecallDataset,
    size: int,
    query_recall_position: int,
) -> Float[Array, " list_length"]:
    """Conditional PNR when study items may repeat.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.
    query_recall_position : int
        0-based recall index to analyze.

    Returns
    -------
    Float[Array, " list_length"]
        Conditional probability at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]

    actual = vmap(actual_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )
    available = vmap(available_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )

    numerator = actual.sum(axis=0)
    denominator = available.sum(axis=0)
    return numerator / denominator


def plot_pnr(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_recall_position: int = 0,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    size: int = 3,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot probability of nth recall with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    query_recall_position : int
        0-based recall index to plot.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
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
        Matplotlib Axes with the PNR plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)


    for data_index, data in enumerate(datasets):
        last_position = last_study_position(data, trial_masks[data_index])
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(
                    conditional_pnr_with_repeats,
                    static_argnames=("size", "query_recall_position"),
                ),
                size=size,
                query_recall_position=query_recall_position,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            jnp.arange(last_position, dtype=int) + 1,
            subject_values[:, :last_position],
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(
        axis,
        "Study Position",
        recall_position_label(query_recall_position),
        contrast_name,
    )
    return axis
