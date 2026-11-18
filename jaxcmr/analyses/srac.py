"""Recall of each study item in its assigned output position.

For the study list A, B, C, D, a participant who recalls B, C and then stops
receives no credit under strict serial scoring. B and C occurred in study
order, but each came one response too early. In relative serial scoring,
the transition from B to C earns credit for C.

``trial_srac`` compares each response with the item studied at the corresponding
position. If an item was presented twice, it can be correct in either of its
assigned slots. ``srac`` averages the resulting indicators across trials,
including trials with a blank at that position. ``plot_srac`` draws the mean
of the participant curves.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject, find_max_list_length

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["plot_srac", "srac", "trial_srac"]


def trial_srac(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Flag each study position as correctly recalled or not.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for one trial; 0 for no recall.
    presentations : Integer[Array, " study_positions"]
        Item IDs in study order.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " study_positions"]
        True where the study position was correctly recalled.

    """
    list_length = presentations.shape[0]
    recalls = recalls[:list_length]
    expanded_recalls = vmap(equivalent_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )
    study_positions = jnp.arange(1, list_length + 1)
    return vmap(lambda r, i: jnp.any(r == i))(expanded_recalls, study_positions)


def srac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Proportion of correct recalls at each study position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean accuracy at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    return vmap(trial_srac, in_axes=(0, 0, None))(
        recalls,
        presentations,
        size,
    ).mean(axis=0)


def plot_srac(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
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
    """Plot serial recall accuracy with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
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
        Matplotlib Axes with the SRAC plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(srac, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        xvals = jnp.arange(max_list_length) + 1
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_mean_curve(
            axis,
            xvals,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(
        axis, "Study Position", "Serial Recall Accuracy", contrast_name
    )
    return axis
