"""Incoming repetition lag-CRP using reversed recall sequences.

Computes backward (incoming) repetition CRP by reversing recall
sequences and delegating to ``repcrp``. Statistical tests compare
observed incoming CRP against shuffled controls and test for
first-vs-second presentation bias.

"""

__all__ = [
    "plot_back_rep_crp",
    "subject_back_rep_crp",
    "BackRepCRPTestResult",
    "test_back_rep_crp_vs_control",
    "test_first_second_bias",
]

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy import stats

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Integer, RecallDataset
from . import repcrp as repcrp_module


def _reverse_nonzero_recalls(
    recalls: Integer[Array, " trial_count recall_events"],
) -> Integer[Array, " trial_count recall_events"]:
    """Reverse recall sequences keeping padding at end.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_events"]
        Recall sequences with zero padding after
        termination.

    """

    def reverse_row(
        row: Integer[Array, " recall_events"],
    ) -> Integer[Array, " recall_events"]:
        count = jnp.sum(row > 0)
        idx = jnp.arange(row.size)
        rev_idx = jnp.clip(count - 1 - idx, 0, row.size - 1)
        reversed_part = row[rev_idx]
        return jnp.where(idx < count, reversed_part, 0)

    return vmap(reverse_row)(recalls)


def plot_back_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot incoming repetition lag-CRP with CIs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to plot.
    min_lag : int, optional
        Minimum separation between repeated
        presentations.
    size : int, optional
        Maximum study positions per item.
    repetition_index : int or None, optional
        Specific repetition index to plot.
    color_cycle : list[str] or None, optional
        Colors for plotting each dataset.
    labels : Sequence[str] or None, optional
        Labels for repetition-index lines.
    contrast_name : str or None, optional
        Legend title.
    axis : Axes or None, optional
        Existing Axes to plot on.
    confidence_level : float, optional
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Axes with the incoming repetition CRP plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    labels_list = list(labels) if labels is not None else []
    if isinstance(repetition_index, int):
        repetition_indices = [repetition_index]
    else:
        repetition_indices = list(range(size))

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        reversed_data = dict(data)
        reversed_data["recalls"] = _reverse_nonzero_recalls(data["recalls"])
        subject_values = apply_by_subject(
            reversed_data,
            trial_masks[data_index],
            jit(repcrp_module.repcrp, static_argnames=("min_lag", "size")),
            min_lag=min_lag,
            size=size,
        )

        for rep_idx, repetition_index in enumerate(repetition_indices):
            repetition_subject_values = jnp.vstack(
                [each[repetition_index] for each in subject_values]
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

            label = (
                labels_list[repetition_index]
                if len(datasets) == 1 and repetition_index < len(labels_list)
                else str(repetition_index + 1)
            )
            color_idx = data_index * len(repetition_indices) + rep_idx
            color = color_cycle[color_idx % len(color_cycle)]
            plot_data(
                axis,
                lag_interval,
                repetition_subject_values,
                label,
                color,
                confidence_level=confidence_level,
            )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis


def subject_back_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> np.ndarray:
    """Compute subject-level incoming repetition CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials to include.
    min_lag : int, optional
        Minimum spacing between item repetitions.
    max_lag : int, optional
        Maximum lag to include in output.
    size : int, optional
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size, 2*max_lag+1)``.

    """
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    reversed_dataset = dict(dataset)
    reversed_dataset["recalls"] = _reverse_nonzero_recalls(dataset["recalls"])
    subject_values = apply_by_subject(
        reversed_dataset,
        trial_mask,
        jit(repcrp_module.repcrp, static_argnames=("min_lag", "size")),
        min_lag=min_lag,
        size=size,
    )
    return np.stack([s[:, lag_slice] for s in subject_values])


@dataclass
class BackRepCRPTestResult:
    """Results from a backward repetition CRP statistical test."""

    lags: np.ndarray
    t_stats: np.ndarray
    t_pvals: np.ndarray
    w_stats: np.ndarray
    w_pvals: np.ndarray
    mean_diffs: np.ndarray

    def __str__(self) -> str:
        lines = [
            f"{'Lag':>5} | {'t-stat':>8} {'t p-val':>10} | "
            f"{'W-stat':>8} {'W p-val':>10} | {'Mean Diff':>10}",
            f"{'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*11}",
        ]
        for i, lag in enumerate(self.lags):
            lines.append(
                f"{lag:>5} | {self.t_stats[i]:>8.3f} {self.t_pvals[i]:>10.4f} | "
                f"{self.w_stats[i]:>8.1f} {self.w_pvals[i]:>10.4f} | "
                f"{self.mean_diffs[i]:>10.4f}"
            )
        return "\n".join(lines)


def test_back_rep_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 5,
) -> dict[str, BackRepCRPTestResult]:
    """Test observed vs control backward CRP per index.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    max_lag : int, optional
        Maximum lag used for labeling.

    Returns
    -------
    dict[str, BackRepCRPTestResult]
        Results keyed by presentation label.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    n_lags = len(lag_labels)
    size = observed_crp.shape[1]
    results = {}

    for rep_idx in range(size):
        obs = observed_crp[:, rep_idx, :]
        ctrl = control_crp[:, rep_idx, :]

        t_stats = np.zeros(n_lags)
        t_pvals = np.zeros(n_lags)
        w_stats = np.zeros(n_lags)
        w_pvals = np.zeros(n_lags)
        mean_diffs = np.zeros(n_lags)

        for lag_idx in range(n_lags):
            obs_col = obs[:, lag_idx]
            ctrl_col = ctrl[:, lag_idx]
            diff = obs_col - ctrl_col

            t_stat, t_pval = stats.ttest_rel(obs_col, ctrl_col, nan_policy="omit")
            t_stats[lag_idx] = t_stat
            t_pvals[lag_idx] = t_pval

            valid = ~(np.isnan(obs_col) | np.isnan(ctrl_col))
            if valid.sum() > 10:
                w_stat, w_pval = stats.wilcoxon(diff[valid], alternative="two-sided")
            else:
                w_stat, w_pval = np.nan, np.nan
            w_stats[lag_idx] = w_stat
            w_pvals[lag_idx] = w_pval
            mean_diffs[lag_idx] = np.nanmean(diff)

        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = BackRepCRPTestResult(
            lags=lag_labels,
            t_stats=t_stats,
            t_pvals=t_pvals,
            w_stats=w_stats,
            w_pvals=w_pvals,
            mean_diffs=mean_diffs,
        )

    return results


def test_first_second_bias(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 5,
) -> BackRepCRPTestResult:
    """Test whether first-presentation bias differs from control.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    max_lag : int, optional
        Maximum lag used for labeling.

    Returns
    -------
    BackRepCRPTestResult
        Test statistics for the difference of differences.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    n_lags = len(lag_labels)

    observed_diff = observed_crp[:, 0, :] - observed_crp[:, 1, :]
    control_diff = control_crp[:, 0, :] - control_crp[:, 1, :]
    effect = observed_diff - control_diff

    t_stats = np.zeros(n_lags)
    t_pvals = np.zeros(n_lags)
    w_stats = np.zeros(n_lags)
    w_pvals = np.zeros(n_lags)
    mean_diffs = np.zeros(n_lags)

    for lag_idx in range(n_lags):
        obs_d = observed_diff[:, lag_idx]
        ctrl_d = control_diff[:, lag_idx]
        diff_of_diff = effect[:, lag_idx]

        t_stat, t_pval = stats.ttest_rel(obs_d, ctrl_d, nan_policy="omit")
        t_stats[lag_idx] = t_stat
        t_pvals[lag_idx] = t_pval

        valid = ~(np.isnan(obs_d) | np.isnan(ctrl_d))
        if valid.sum() > 10:
            w_stat, w_pval = stats.wilcoxon(diff_of_diff[valid], alternative="two-sided")
        else:
            w_stat, w_pval = np.nan, np.nan
        w_stats[lag_idx] = w_stat
        w_pvals[lag_idx] = w_pval
        mean_diffs[lag_idx] = np.nanmean(diff_of_diff)

    return BackRepCRPTestResult(
        lags=lag_labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
    )
