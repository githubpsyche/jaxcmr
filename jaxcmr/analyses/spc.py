"""Recall probability by study position.

The serial position curve (SPC) shows how often participants recall the item
studied at each position, wherever it appears in the response sequence. On a
trial with study list A, B, C and recalls C, A, positions 1 and 3 each receive
one count. If the same item appeared at several study positions, recalling it
once credits each of those positions. Recalling it again adds no further credit.

``spc`` divides the count at each position by the number of trials with an item
at that position. ``plot_spc`` calculates this curve separately for each
participant and plots their mean. For fixed-length lists of distinct items,
``fixed_pres_spc`` computes the curve directly from position-coded recalls.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit
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

__all__ = ["fixed_pres_spc", "plot_spc", "spc"]


def fixed_pres_spc(
    recalls: Integer[Array, " trial_count recall_positions"],
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Compute recall rate by study position for fixed-length lists.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_positions"]
        Trial by recall position array of recalled items (1-indexed; 0 = no recall).
    list_length : int
        Number of items in the study list.

    Returns
    -------
    Float[Array, " study_positions"]
        Recall probability at each serial position.

    """
    return recalled_positions(recalls, list_length).mean(axis=0)


def spc(dataset: RecallDataset) -> Float[Array, " study_positions"]:
    """Return the probability of recalling the item at each study position.

    Parameters
    ----------
    dataset : RecallDataset
        Study lists, lengths and position-coded responses. Recalling an item
        credits all its presentations; duplicate responses count once.

    Returns
    -------
    array of shape (study_positions,)
        Recalled presentations divided by eligible presentations at each
        position. Padding is excluded; positions with no opportunities are NaN.
        Trials are pooled here; ``plot_spc`` averages participant probabilities.
    """
    return recall_status(dataset).sum(axis=0) / study_position_mask(dataset).sum(axis=0)


def plot_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot serial position curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets. Each is paired with the mask at the same index.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials. Length must match ``datasets``.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels for each curve.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Axes to plot on.
    confidence_level : float, optional
        Confidence level for error bounds (default: 0.95).

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with serial position curves.
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
                jit(spc),
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

    format_analysis_axes(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
