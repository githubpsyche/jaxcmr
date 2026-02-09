"""Backward repetition lag-rank temporal factor score.

Computes per-presentation temporal factor scores for incoming
transitions TO repeated items by reversing recall sequences and
delegating to ``replagrank``. Mirrors the ``cleanbackrepcrp``
wrapper pattern.

"""

__all__ = [
    "backreplagrank",
    "subject_back_rep_lagrank",
    "plot_back_rep_lagrank",
]

from typing import Optional, Sequence

import numpy as np
from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, Integer, RecallDataset
from .cleanbackrepcrp import _reverse_nonzero_recalls
from .replagrank import (
    RepLagRankTestResult,
    plot_rep_lagrank,
    replagrank,
    subject_rep_lagrank,
    test_first_second_bias,
    test_rep_lagrank_vs_control,
)


def backreplagrank(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, "trials size"]:
    """Compute per-trial per-presentation incoming lag-rank factors.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    Returns
    -------
    Float[Array, "trials size"]
        Per-trial temporal factor for each presentation index.

    """
    reversed_data = dict(dataset)
    reversed_data["recalls"] = _reverse_nonzero_recalls(dataset["recalls"])
    return replagrank(reversed_data, min_lag=min_lag, size=size)


def subject_back_rep_lagrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    size: int = 2,
) -> np.ndarray:
    """Compute per-subject per-presentation incoming lag-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size)``.

    """
    reversed_dataset = dict(dataset)
    reversed_dataset["recalls"] = _reverse_nonzero_recalls(dataset["recalls"])
    return subject_rep_lagrank(
        reversed_dataset, trial_mask, min_lag=min_lag, size=size
    )


def plot_back_rep_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot per-presentation incoming lag-rank factors.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.
    color_cycle : list[str] or None
        Colors for each bar.
    labels : Sequence[str] or None
        Labels for each presentation bar.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the grouped bar chart.

    """
    if not isinstance(datasets, Sequence) or isinstance(datasets, dict):
        datasets = [datasets]
    if not isinstance(trial_masks, Sequence) or isinstance(trial_masks, np.ndarray):
        if hasattr(trial_masks, 'ndim') and trial_masks.ndim == 1:
            trial_masks = [trial_masks]

    reversed_datasets = []
    for data in datasets:
        rd = dict(data)
        rd["recalls"] = _reverse_nonzero_recalls(data["recalls"])
        reversed_datasets.append(rd)

    return plot_rep_lagrank(
        reversed_datasets,
        trial_masks,
        min_lag=min_lag,
        size=size,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )
