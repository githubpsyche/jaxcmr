"""Strict serial accuracy for successive presentations of a repeated item.

Suppose a participant studies A, X, B, X and recalls A, X before stopping.
The first presentation of X was recalled in its assigned output slot. The
second was missed. This trial contributes one correct response out of one
opportunity for the first presentation, and zero out of one for the second.

``trial_repsrac_counts`` groups the position scores from ``trial_srac`` by
occurrence number for items studied more than once. ``repsrac`` pools correct
responses and opportunities across trials before dividing. An occurrence
number with no opportunities receives a score of zero.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset
from .srac import trial_srac

__all__ = ["plot_repsrac", "repsrac", "trial_repsrac_counts"]


def trial_repsrac_counts(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> tuple[Integer[Array, " repetition_index"], Integer[Array, " repetition_index"]]:
    """Return correct and total counts per repetition index.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        Recalled item indices for one trial (1-indexed,
        0 = no recall).
    presentations : Integer[Array, " study_positions"]
        Item identifiers in study order (1-indexed).
    size : int, optional
        Maximum study positions per item.

    Returns
    -------
    tuple of Integer[Array, " repetition_index"]
        Correct counts and total counts per repetition
        index.

    """
    list_length = presentations.shape[0]
    correct_positions = trial_srac(recalls, presentations, size)
    study_positions = jnp.arange(1, list_length + 1)
    item_positions = vmap(equivalent_study_positions, in_axes=(0, None, None))(
        study_positions, presentations, size
    )
    repeated_mask = (
        item_positions[:, 1] > 0
        if size > 1
        else jnp.zeros_like(study_positions, dtype=bool)
    )
    matches = item_positions == study_positions[:, None]
    index_mask = matches & repeated_mask[:, None]
    correct_counts = jnp.sum(index_mask & correct_positions[:, None], axis=0)
    total_counts = jnp.sum(index_mask, axis=0)
    return correct_counts, total_counts


def repsrac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " repetition_index"]:
    """Return recall accuracy for each repetition index.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and
        ``pres_itemnos``.
    size : int, optional
        Maximum study positions per item.

    Returns
    -------
    Float[Array, " repetition_index"]
        Accuracy for each repetition index.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    correct_counts, total_counts = vmap(trial_repsrac_counts, in_axes=(0, 0, None))(
        recalls, presentations, size
    )
    total_correct = correct_counts.sum(axis=0)
    total_possible = total_counts.sum(axis=0)
    return jnp.where(total_possible > 0, total_correct / total_possible, 0.0)


def plot_repsrac(
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
    """Plot repetition-index serial recall accuracy.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials for each dataset.
    color_cycle : list[str] or None, optional
        Line colors for each dataset.
    labels : Sequence[str] or None, optional
        Legend entries for each dataset.
    contrast_name : str or None, optional
        Legend title.
    axis : Axes or None, optional
        Existing Axes to draw on.
    size : int, optional
        Maximum study positions per item.
    confidence_level : float, optional
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Axes containing the plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = ["" for _ in datasets]

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(repsrac, static_argnames=("size",)),
                size=size,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
        xvals = jnp.arange(size) + 1
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
        axis, "Repetition Index", "Serial Recall Accuracy", contrast_name
    )
    return axis
