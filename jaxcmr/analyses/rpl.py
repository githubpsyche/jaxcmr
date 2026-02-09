"""Recall Probability by Lag (RPL).

Computes recall probability as a function of the lag between a
repeated item's two study presentations. Quantifies how spacing
between presentations affects subsequent memory.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes
from scipy import stats

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

__all__ = [
    "infer_max_lag",
    "item_lag_counts",
    "recall_probability_by_lag",
    "binned_recall_probability_by_lag",
    "plot_full_rpl",
    "plot_rpl",
    "subject_full_rpl",
    "subject_binned_rpl",
    "test_rpl_slope",
    "test_rpl_slope_vs_comparison",
    "run_rpl_slope_analysis",
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
    """One-hot presented/recalled vectors for an item's lag bin.

    Parameters
    ----------
    target_item : Int_
        Item ID to tabulate.
    recalls : Integer[Array, " recall_events"]
        1-indexed recalled positions; 0 for no recall.
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    max_lag : int
        Largest explicit lag bucket.
    n_bins : int
        Total number of bins (``max_lag + 2``).

    Returns
    -------
    tuple[Bool[Array, " lag_bins"], Bool[Array, " lag_bins"]]
        Presented and recalled one-hot vectors.

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
    dataset: RecallDataset,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Recall probability by repetition lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    max_lag : int
        Largest explicit lag bucket.

    Returns
    -------
    Float[Array, " lag_bins"]
        Recall probability per lag bin.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]
    n_bins = max_lag + 2
    all_items = jnp.arange(1, list_length + 1)

    presented_t, recalled_t = vmap(_trial_lag_counts, in_axes=(0, 0, None, None, None))(
        recalls, presentations, all_items, max_lag, n_bins
    )
    return recalled_t.sum(0) / presented_t.sum(0)


def binned_recall_probability_by_lag(
    dataset: RecallDataset,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Binned recall probability by repetition lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    max_lag : int
        Largest explicit lag bucket.

    Returns
    -------
    Float[Array, " lag_bins"]
        Recall probability per binned lag.

    """
    result = recall_probability_by_lag(dataset, max_lag)
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
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot full-resolution repetition-lag curves.

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
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the RPL plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)
    max_lag = infer_max_lag(
        datasets[0]["pres_itemnos"], datasets[0]["pres_itemnos"].shape[1]
    )
    xticklabels = ["N/A"] + [f"{i}" for i in range(max_lag + 1)]
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(recall_probability_by_lag, static_argnames=("max_lag",)),
                max_lag,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            jnp.arange(len(xticklabels)),
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    axis.set_xticks(range(len(xticklabels)))
    axis.set_xticklabels(xticklabels)
    set_plot_labels(axis, "Lag", "Recall Rate", contrast_name)
    return axis


def plot_rpl(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot binned repetition-lag curves.

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
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the binned RPL plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)
    max_lag = infer_max_lag(
        datasets[0]["pres_itemnos"], datasets[0]["pres_itemnos"].shape[1]
    )
    xticklabels = ["N/A", "0", "1-2", "3-5", "6-8"]
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(binned_recall_probability_by_lag, static_argnames=("max_lag",)),
                max_lag,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            jnp.arange(5),
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    axis.set_xticks(range(len(xticklabels)))
    axis.set_xticklabels(xticklabels)
    set_plot_labels(axis, "Lag", "Recall Rate", contrast_name)
    return axis


_BINNED_LAG_VALUES = np.array([0.0, 1.5, 4.0, 7.0])


def _resolve_max_lag(
    dataset: RecallDataset,
    max_lag: int | None,
) -> int:
    """Return the explicit lag bucket limit.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset used to infer the limit when ``max_lag`` is None.
    max_lag : int or None
        Explicit lag bucket limit if provided.

    Returns
    -------
    int
        Resolved lag bucket limit.

    """
    if max_lag is not None:
        return max_lag
    return infer_max_lag(dataset["pres_itemnos"], dataset["pres_itemnos"].shape[1])


def subject_full_rpl(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int | None = None,
) -> np.ndarray:
    """Subject-level recall probability by lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    max_lag : int or None
        Largest explicit lag bucket.

    Returns
    -------
    np.ndarray
        Shape [n_subjects, max_lag + 2].

    """
    max_lag = _resolve_max_lag(dataset, max_lag)
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(recall_probability_by_lag, static_argnames=("max_lag",)),
        max_lag,
    )
    return np.vstack(subject_values)


def subject_binned_rpl(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int | None = None,
) -> np.ndarray:
    """Subject-level binned recall probability by lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    max_lag : int or None
        Largest explicit lag bucket.

    Returns
    -------
    np.ndarray
        Shape [n_subjects, 5].

    """
    max_lag = _resolve_max_lag(dataset, max_lag)
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(binned_recall_probability_by_lag, static_argnames=("max_lag",)),
        max_lag,
    )
    return np.vstack(subject_values)


def _lag_values_for_mode(mode: str, n_bins: int) -> np.ndarray:
    """Return lag values for slope fits.

    Parameters
    ----------
    mode : str
        ``"full"`` or ``"binned"``.
    n_bins : int
        Number of lag bins excluding single-presentation bin.

    Returns
    -------
    np.ndarray
        Lag values aligned to the slope-fit columns.

    """
    if mode == "full":
        return np.arange(n_bins, dtype=float)
    if mode == "binned":
        if n_bins != _BINNED_LAG_VALUES.size:
            raise ValueError("Binned mode expects four lag bins.")
        return _BINNED_LAG_VALUES.copy()
    raise ValueError("Mode must be 'full' or 'binned'.")


def _fit_subject_slopes(
    subject_values: np.ndarray,
    lag_values: np.ndarray,
) -> np.ndarray:
    """Per-subject slope estimates for recall probability by lag.

    Parameters
    ----------
    subject_values : np.ndarray
        Subject-level recall probabilities for each lag.
    lag_values : np.ndarray
        Lag values aligned to columns in ``subject_values``.

    Returns
    -------
    np.ndarray
        Shape ``[n_subjects]``; NaN where insufficient data.

    """
    slopes = np.full(subject_values.shape[0], np.nan)
    for idx, values in enumerate(subject_values):
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        slopes[idx] = np.polyfit(lag_values[valid], values[valid], 1)[0]
    return slopes


@dataclass
class RPLSlopeTestResult:
    """Results from a recall-probability-by-lag slope test."""

    n: int
    mean_slope: float
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float

    def __str__(self) -> str:
        lines = [
            f"N={self.n}",
            f"Mean slope: {self.mean_slope:.4f}",
            f"t-stat: {self.t_stat:.3f} p={self.t_pval:.4f}",
            f"W-stat: {self.w_stat:.1f} p={self.w_pval:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class RPLSlopeComparisonResult:
    """Results from a comparison of recall-probability-by-lag slopes."""

    n: int
    mean_observed: float
    mean_comparison: float
    mean_diff: float
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float

    def __str__(self) -> str:
        lines = [
            f"N={self.n}",
            f"Mean slope (observed): {self.mean_observed:.4f}",
            f"Mean slope (comparison): {self.mean_comparison:.4f}",
            f"Mean difference: {self.mean_diff:.4f}",
            f"t-stat: {self.t_stat:.3f} p={self.t_pval:.4f}",
            f"W-stat: {self.w_stat:.1f} p={self.w_pval:.4f}",
        ]
        return "\n".join(lines)


def test_rpl_slope(
    subject_rpl: np.ndarray,
    mode: str = "full",
) -> RPLSlopeTestResult:
    """Test whether recall probability increases with spacing.

    Parameters
    ----------
    subject_rpl : np.ndarray
        Subject-level recall probabilities by lag.
    mode : str
        ``"full"`` or ``"binned"``.

    Returns
    -------
    RPLSlopeTestResult
        Test statistics for per-subject slopes.

    """
    values = subject_rpl[:, 1:]
    lag_values = _lag_values_for_mode(mode, values.shape[1])
    slopes = _fit_subject_slopes(values, lag_values)

    valid = np.isfinite(slopes)
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_1samp(slopes, 0.0, nan_policy="omit")
    if n > 10:
        w_stat, w_pval = stats.wilcoxon(slopes[valid], alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return RPLSlopeTestResult(
        n=n,
        mean_slope=float(np.nanmean(slopes)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


def test_rpl_slope_vs_comparison(
    observed_rpl: np.ndarray,
    comparison_rpl: np.ndarray,
    mode: str = "full",
) -> RPLSlopeComparisonResult:
    """Test whether observed and comparison slopes differ.

    Parameters
    ----------
    observed_rpl : np.ndarray
        Subject-level recall probabilities (observed).
    comparison_rpl : np.ndarray
        Subject-level recall probabilities (comparison).
    mode : str
        ``"full"`` or ``"binned"``.

    Returns
    -------
    RPLSlopeComparisonResult
        Test statistics for slope differences.

    """
    obs_values = observed_rpl[:, 1:]
    comparison_values = comparison_rpl[:, 1:]
    if obs_values.shape != comparison_values.shape:
        raise ValueError(
            "Observed and comparison RPL arrays must have the same shape."
        )

    lag_values = _lag_values_for_mode(mode, obs_values.shape[1])
    obs_slopes = _fit_subject_slopes(obs_values, lag_values)
    comparison_slopes = _fit_subject_slopes(comparison_values, lag_values)

    valid = ~(np.isnan(obs_slopes) | np.isnan(comparison_slopes))
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_rel(
        obs_slopes, comparison_slopes, nan_policy="omit"
    )
    if n > 10:
        diff = obs_slopes[valid] - comparison_slopes[valid]
        w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return RPLSlopeComparisonResult(
        n=n,
        mean_observed=float(np.nanmean(obs_slopes)),
        mean_comparison=float(np.nanmean(comparison_slopes)),
        mean_diff=float(np.nanmean(obs_slopes - comparison_slopes)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


def run_rpl_slope_analysis(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    comparison_dataset: RecallDataset,
    comparison_mask: Bool[Array, " trial_count"],
    mode: str = "full",
    max_lag: int | None = None,
) -> tuple[RPLSlopeTestResult, RPLSlopeTestResult, RPLSlopeComparisonResult]:
    """Slope tests for observed and comparison datasets.

    Parameters
    ----------
    dataset : RecallDataset
        Observed recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting observed trials.
    comparison_dataset : RecallDataset
        Comparison recall dataset.
    comparison_mask : Bool[Array, " trial_count"]
        Mask selecting comparison trials.
    mode : str
        ``"full"`` or ``"binned"``.
    max_lag : int or None
        Largest explicit lag bucket.

    Returns
    -------
    tuple[RPLSlopeTestResult, RPLSlopeTestResult, RPLSlopeComparisonResult]
        Observed, comparison, and difference results.

    """
    max_lag = _resolve_max_lag(dataset, max_lag)
    if mode == "full":
        observed_rpl = subject_full_rpl(dataset, trial_mask, max_lag)
        comparison_rpl = subject_full_rpl(comparison_dataset, comparison_mask, max_lag)
    elif mode == "binned":
        observed_rpl = subject_binned_rpl(dataset, trial_mask, max_lag)
        comparison_rpl = subject_binned_rpl(
            comparison_dataset, comparison_mask, max_lag
        )
    else:
        raise ValueError("Mode must be 'full' or 'binned'.")

    observed_result = test_rpl_slope(observed_rpl, mode=mode)
    comparison_result = test_rpl_slope(comparison_rpl, mode=mode)
    difference_result = test_rpl_slope_vs_comparison(
        observed_rpl, comparison_rpl, mode=mode
    )

    return observed_result, comparison_result, difference_result
