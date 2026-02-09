"""Conditional co-recall probability by study lag.

Given that an item was recalled, computes the probability that the
item at each study lag was also recalled. Produces a co-recall curve
as a function of absolute study lag.

"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["conditional_corec_by_lag", "plot_conditional_corec_by_lag"]


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


def _available_anchor_counts(
    mask: Float[Array, " study_positions"],
) -> Float[Array, " lags"]:
    """Return anchor counts for each positive lag.

    Parameters
    ----------
    mask : Float[Array, " study_positions"]
        Float mask indicating recalled study positions.

    """

    total_recalled = mask.sum()
    suffix_counts = jnp.cumsum(mask[::-1])[:-1]
    return total_recalled - suffix_counts


def _conditional_corec_counts(
    recalls: Integer[Array, " recall_events"],
    list_length: int,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Return conditional co-recall counts for a trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_events"]
        One-indexed recall positions for a single
        trial (0 pads).
    list_length : int
        Number of studied items in the list.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Positive-lag co-recall counts and the number
        of recalled anchors contributing to each lag.

    """

    mask = _trial_recall_mask(recalls, list_length).astype(jnp.float32)
    corr = jnp.correlate(mask, mask, mode="full")
    actual_pairs = corr[list_length:]
    anchor_counts = _available_anchor_counts(mask)
    return actual_pairs, anchor_counts


def conditional_corec_by_lag(dataset: RecallDataset) -> Float[Array, " lags"]:
    """Return conditional CoRec(d) for positive lags.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and
        ``pres_itemnos``.

    Returns
    -------
    Float[Array, " lags"]
        Conditional co-recall probability at each
        positive lag.

    """

    list_length = dataset["pres_itemnos"].shape[1]
    actual, anchors = vmap(_conditional_corec_counts, in_axes=(0, None))(
        dataset["recalls"], list_length
    )
    return actual.sum(0) / anchors.sum(0)


def plot_conditional_corec_by_lag(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot conditional co-recall curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset or list of datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials within each
        dataset.
    max_lag : int, optional
        Maximum lag to display (defaults to full range).
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
        curves.

    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    for index, data in enumerate(datasets):
        lag_count = data["pres_itemnos"].shape[1] - 1
        lag_limit = lag_count if max_lag is None else min(max_lag, lag_count)
        lag_axis = jnp.arange(1, lag_limit + 1, dtype=int)
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                jit(conditional_corec_by_lag),
            )
        )
        subject_values = subject_values[:, :lag_limit]

        color = color_cycle[index % len(color_cycle)]
        plot_data(
            axis,
            lag_axis,
            subject_values,
            labels[index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Lag", "Conditional Co-Recall Probability", contrast_name)
    return axis
