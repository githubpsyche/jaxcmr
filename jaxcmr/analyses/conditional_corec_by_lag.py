"""Conditional co-recall probability as a function of study lag.

Estimates the probability that a neighbor ``lag`` positions away was recalled
given that the anchor position was recalled, ignoring recall order.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["conditional_corec_by_lag", "plot_conditional_corec_by_lag"]


def _trial_recall_mask(
    recalls: Integer[Array, " recall_events"],
    list_length: int,
) -> Bool[Array, " study_positions"]:
    """Returns a boolean mask indicating which study positions were recalled.

    Args:
      recalls: One-indexed recall positions for a single trial (0 pads).
      list_length: Number of studied items in the list.
    """

    counts = jnp.bincount(recalls, length=list_length + 1)[1:]
    return counts > 0


def _available_anchor_counts(
    mask: Float[Array, " study_positions"],
) -> Float[Array, " lags"]:
    """Returns anchor counts for each positive lag.

    Args:
      mask: Float mask indicating recalled study positions.
    """

    total_recalled = mask.sum()
    suffix_counts = jnp.cumsum(mask[::-1])[:-1]
    return total_recalled - suffix_counts


def _conditional_corec_counts(
    recalls: Integer[Array, " recall_events"],
    list_length: int,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Returns conditional co-recall numerators and denominators for a trial.

    Args:
      recalls: One-indexed recall positions for a single trial (0 pads).
      list_length: Number of studied items in the list.

    Returns:
      (actual_pairs, anchor_counts): Positive-lag co-recall counts and the number
      of recalled anchors contributing to each lag.
    """

    mask = _trial_recall_mask(recalls, list_length).astype(jnp.float32)
    corr = jnp.correlate(mask, mask, mode="full")
    actual_pairs = corr[list_length:]
    anchor_counts = _available_anchor_counts(mask)
    return actual_pairs, anchor_counts


def conditional_corec_by_lag(dataset: RecallDataset) -> Float[Array, " lags"]:
    """Returns conditional CoRec(d) for positive lags.

    Args:
      dataset: Recall dataset containing ``recalls`` and ``pres_itemnos``.
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
) -> Axes:
    """Returns Matplotlib ``Axes`` with conditional co-recall curves.

    Args:
      datasets: Dataset or list of datasets to plot.
      trial_masks: Boolean masks selecting trials within each dataset.
      max_lag: Maximum lag to display (defaults to full range).
      color_cycle: Colors for plotting each dataset.
      labels: Legend labels for each dataset.
      contrast_name: Legend title for contrasts.
      axis: Existing Matplotlib ``Axes`` to plot on.
    """

    axis = init_plot(axis)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]
    if not isinstance(datasets, Sequence):
        datasets = [datasets]
    if not isinstance(trial_masks, Sequence):
        trial_masks = [jnp.array(trial_masks)]
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

        color = color_cycle.pop(0)
        plot_data(
            axis,
            lag_axis,
            subject_values,
            labels[index],
            color,
        )

    set_plot_labels(axis, "Lag", "Conditional Co-Recall Probability", contrast_name)
    return axis
