"""Termination probability curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
from jax import jit
from matplotlib.axes import Axes
from scipy import stats

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = [
    "simple_termination_probability",
    "conditional_termination_probability",
    "plot_termination_probability",
    "subject_output_length_mean",
    "subject_output_length_median",
    "test_output_length_mean_vs_control",
    "test_output_length_median_vs_control",
]


def simple_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Termination probability by recall position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls``.

    Returns
    -------
    Float[Array, " recall_positions"]
        Probability of stopping at each output position.

    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    counts = jnp.bincount(termination_index, length=recalls.shape[1])
    return counts / recalls.shape[0]


def conditional_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Conditional termination probability by recall position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls``.

    Returns
    -------
    Float[Array, " recall_positions"]
        P(stop | reached) at each output position.

    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    stop_counts = jnp.bincount(termination_index, length=recalls.shape[1])
    recall_positions = jnp.arange(recalls.shape[1])
    reached_mask = recall_positions <= termination_index[:, None]
    reached_counts = jnp.sum(reached_mask, axis=0)
    return stop_counts / reached_counts


def plot_termination_probability(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    mode: str = "conditional",
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot termination probability curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    mode : str
        ``"conditional"`` or ``"simple"``.
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
        Matplotlib Axes with termination curves.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    max_recall_length = max(
        int(dataset["recalls"].shape[1])
        for dataset, mask in zip(datasets, trial_masks)
        if jnp.any(mask)
    )

    if mode == "conditional":
        curve_fn = jit(conditional_termination_probability)
    elif mode == "simple":
        curve_fn = jit(simple_termination_probability)

    for index, data in enumerate(datasets):
        subject_curves = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                curve_fn,
            )
        )

        subject_curves = subject_curves[:, :max_recall_length]
        recall_positions = jnp.arange(max_recall_length, dtype=jnp.int32) + 1

        color = color_cycle[index % len(color_cycle)]
        plot_data(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
        )

    ylabel = "P(Terminate | Reach)" if mode == "conditional" else "P(Terminate)"
    set_plot_labels(axis, "Recall Position", ylabel, contrast_name)
    return axis


def _output_lengths(
    dataset: RecallDataset,
) -> Integer[Array, " trial_count"]:
    """Return output lengths per trial."""
    recalls = dataset["recalls"]
    return jnp.sum(recalls > 0, axis=1)


def _mean_output_length(
    dataset: RecallDataset,
) -> Float[Array, ""]:
    """Return the mean output length for a dataset slice."""
    return jnp.mean(_output_lengths(dataset))


def _median_output_length(
    dataset: RecallDataset,
) -> Float[Array, ""]:
    """Return the median output length for a dataset slice."""
    return jnp.median(_output_lengths(dataset))


def subject_output_length_mean(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
) -> np.ndarray:
    """Mean output lengths per subject.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.

    Returns
    -------
    np.ndarray
        Per-subject mean output lengths.

    """
    subject_values = apply_by_subject(dataset, trial_mask, _mean_output_length)
    return np.asarray(subject_values, dtype=float)


def subject_output_length_median(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
) -> np.ndarray:
    """Median output lengths per subject.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.

    Returns
    -------
    np.ndarray
        Per-subject median output lengths.

    """
    subject_values = apply_by_subject(dataset, trial_mask, _median_output_length)
    return np.asarray(subject_values, dtype=float)


@dataclass
class OutputLengthTestResult:
    """Results from an output-length statistical test."""

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
            f"Mean (observed): {self.mean_observed:.4f}",
            f"Mean (control): {self.mean_control:.4f}",
            f"Mean difference: {self.mean_diff:.4f}",
            f"t-stat: {self.t_stat:.3f} p={self.t_pval:.4f}",
            f"W-stat: {self.w_stat:.1f} p={self.w_pval:.4f}",
        ]
        return "\n".join(lines)


def _test_output_lengths(
    observed: np.ndarray,
    control: np.ndarray,
) -> OutputLengthTestResult:
    """Return paired tests for per-subject output length summaries."""
    if observed.shape != control.shape:
        raise ValueError("Observed and control arrays must have the same shape.")

    valid = ~(np.isnan(observed) | np.isnan(control))
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_rel(observed, control, nan_policy="omit")
    if n > 10:
        diff = observed[valid] - control[valid]
        w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return OutputLengthTestResult(
        n=n,
        mean_observed=float(np.nanmean(observed)),
        mean_control=float(np.nanmean(control)),
        mean_diff=float(np.nanmean(observed - control)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


def test_output_length_mean_vs_control(
    observed_means: np.ndarray,
    control_means: np.ndarray,
) -> OutputLengthTestResult:
    """Test mean output length: observed vs control.

    Parameters
    ----------
    observed_means : np.ndarray
        Per-subject mean output lengths (observed).
    control_means : np.ndarray
        Per-subject mean output lengths (control).

    Returns
    -------
    OutputLengthTestResult
        Paired test statistics.

    """
    return _test_output_lengths(observed_means, control_means)


def test_output_length_median_vs_control(
    observed_medians: np.ndarray,
    control_medians: np.ndarray,
) -> OutputLengthTestResult:
    """Test median output length: observed vs control.

    Parameters
    ----------
    observed_medians : np.ndarray
        Per-subject median output lengths (observed).
    control_medians : np.ndarray
        Per-subject median output lengths (control).

    Returns
    -------
    OutputLengthTestResult
        Paired test statistics.

    """
    return _test_output_lengths(observed_medians, control_medians)
