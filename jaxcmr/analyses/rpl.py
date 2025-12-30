"""Recall probability by lag.

Compute how recall varies with the spacing between repeated item presentations and
provide plotting utilities for visualizing repetition lag effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from scipy import stats

from ..helpers import apply_by_subject
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
    "subject_full_rpl",
    "subject_binned_rpl",
    "test_rpl_slope",
    "test_rpl_slope_vs_control",
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
    dataset: RecallDataset,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Return recall probability for lag 0, 1, ..., ``max_lag``, plus once-presented items.

    Args:
        dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.
        max_lag: Largest explicit lag bucket.
            - bin 0 → single presentation
            - bin k → k intervening items for 1 ≤ k ≤ ``max_lag``
            - bin ``max_lag + 1`` → lag exceeds ``max_lag``
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
    """Return binned recall probability for lag 0, 1, ..., ``max_lag``, plus once-presented items.

    Args:
        dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.
        max_lag: Largest explicit lag bucket.
            - bin 0 → single presentation
            - bin k → k intervening items for 1 ≤ k ≤ ``max_lag``
            - bin ``max_lag + 1`` → lag exceeds ``max_lag``
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
) -> Axes:
    """Return ``Axes`` with repetition-lag curves for each dataset and mask.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Boolean masks to filter trials.
        color_cycle: Colors for plotting each dataset.
        labels: Names for each dataset for the legend.
        contrast_name: Legend title for the plotted contrast.
        axis: Existing Matplotlib axes to plot on.
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

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(len(xticklabels)),
            subject_values,
            labels[data_index],
            color,
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
) -> Axes:
    """Return ``Axes`` with binned repetition-lag curves for each dataset and mask.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Boolean masks to filter trials.
        color_cycle: Colors for plotting each dataset.
        labels: Names for each dataset for the legend.
        contrast_name: Legend title for the plotted contrast.
        axis: Existing Matplotlib axes to plot on.
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

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(5),
            subject_values,
            labels[data_index],
            color,
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
    """Returns the explicit lag bucket limit.

    Args:
        dataset: Dataset used to infer the lag limit when ``max_lag`` is None.
        max_lag: Explicit lag bucket limit to use when provided.
    """
    if max_lag is not None:
        return max_lag
    return infer_max_lag(dataset["pres_itemnos"], dataset["pres_itemnos"].shape[1])


def subject_full_rpl(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int | None = None,
) -> np.ndarray:
    """Compute subject-level recall probability by lag.

    Args:
        dataset: Recall dataset containing trial data.
        trial_mask: Boolean mask selecting trials to include.
        max_lag: Largest explicit lag bucket to use.

    Returns:
        Array of shape [n_subjects, max_lag + 2] with recall probabilities by lag.
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
    """Compute subject-level binned recall probability by lag.

    Args:
        dataset: Recall dataset containing trial data.
        trial_mask: Boolean mask selecting trials to include.
        max_lag: Largest explicit lag bucket to use.

    Returns:
        Array of shape [n_subjects, 5] with binned recall probabilities by lag.
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

    Args:
        mode: "full" or "binned".
        n_bins: Number of lag bins excluding the single-presentation bin.

    Returns:
        Array of lag values aligned to the slope-fit columns.
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
    """Return per-subject slope estimates for recall probability by lag.

    Args:
        subject_values: Subject-level recall probabilities for each lag.
        lag_values: Lag values aligned to the columns in ``subject_values``.

    Returns:
        Array of shape [n_subjects] with slope estimates; NaN when insufficient data.
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
    mean_control: float
    mean_diff: float
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float

    def __str__(self) -> str:
        lines = [
            f"N={self.n}",
            f"Mean slope (observed): {self.mean_observed:.4f}",
            f"Mean slope (control): {self.mean_control:.4f}",
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

    Fits a slope for each subject across lag bins (excluding the
    single-presentation bin) and tests whether the mean slope differs from zero.

    Args:
        subject_rpl: Subject-level recall probabilities by lag.
        mode: "full" or "binned".

    Returns:
        RPLSlopeTestResult with test statistics for the per-subject slopes.
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


def test_rpl_slope_vs_control(
    observed_rpl: np.ndarray,
    control_rpl: np.ndarray,
    mode: str = "full",
) -> RPLSlopeComparisonResult:
    """Test whether observed and control spacing slopes differ.

    Fits per-subject slopes across lag bins (excluding the single-presentation bin)
    and compares observed and control slopes.

    Args:
        observed_rpl: Subject-level recall probabilities by lag from observed data.
        control_rpl: Subject-level recall probabilities by lag from control data.
        mode: "full" or "binned".

    Returns:
        RPLSlopeComparisonResult with test statistics for slope differences.
    """
    obs_values = observed_rpl[:, 1:]
    ctrl_values = control_rpl[:, 1:]
    if obs_values.shape != ctrl_values.shape:
        raise ValueError("Observed and control RPL arrays must have the same shape.")

    lag_values = _lag_values_for_mode(mode, obs_values.shape[1])
    obs_slopes = _fit_subject_slopes(obs_values, lag_values)
    ctrl_slopes = _fit_subject_slopes(ctrl_values, lag_values)

    valid = ~(np.isnan(obs_slopes) | np.isnan(ctrl_slopes))
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_rel(obs_slopes, ctrl_slopes, nan_policy="omit")
    if n > 10:
        diff = obs_slopes[valid] - ctrl_slopes[valid]
        w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return RPLSlopeComparisonResult(
        n=n,
        mean_observed=float(np.nanmean(obs_slopes)),
        mean_control=float(np.nanmean(ctrl_slopes)),
        mean_diff=float(np.nanmean(obs_slopes - ctrl_slopes)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )

