from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import bootstrap

from jaxcmr.typing import Array, Bool, Integer, Real

__all__ = [
    "init_plot",
    "plot_data",
    "plot_ratio_data",
    "calculate_errors",
    "plot_with_error_bars",
    "plot_without_error_bars",
    "set_plot_labels",
]

BootstrapMode = Literal["omit", "subjectwise", "trialwise", "hierarchical"]


def segment_by_nan(vector: jnp.ndarray) -> list[tuple[int, int]]:
    """Return index ranges split by NaN entries.

    Args:
        vector: Vector to segment.

    Returns:
        List of (start, end) index pairs.
    """
    segments = []
    start_idx = 0
    for i in range(len(vector)):
        if jnp.isnan(vector[i]):
            segments.append((start_idx, i))
            start_idx = i + 1
    if start_idx < len(vector):
        segments.append((start_idx, len(vector)))
    return segments


def init_plot(axis: Optional[Axes] = None) -> Axes:
    """Return an initialized plotting axis.

    Args:
        axis: An existing axis to use for the plot.
    """
    if axis is None:
        plt.figure()
        axis = plt.gca()
    return axis


def calculate_errors(
    y_values: jnp.ndarray,
    y_mean: jnp.ndarray,
    *,
    confidence_level: float = 0.95,
) -> jnp.ndarray:
    """Return error bars from bootstrapped confidence intervals.

    Args:
        y_values: Array of y values.
        y_mean: Array of y mean values.
        confidence_level: Confidence level for the interval.
    """
    errors = jnp.zeros((2, len(y_mean)))
    bootstrapped_confidence_intervals = bootstrap(
        (y_values,), jnp.nanmean, confidence_level=confidence_level
    ).confidence_interval

    return (
        errors.at[0]
        .set(y_mean - bootstrapped_confidence_intervals.low)
        .at[1]
        .set(bootstrapped_confidence_intervals.high - y_mean)
    )


def plot_with_error_bars(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_values: Real[Array, " trial_count values"],
    y_mean: Real[Array, " values"],
    segments: list[tuple[int, int]],
    label: str,
    color: str,
    *,
    confidence_level: float = 0.95,
) -> None:
    """Plot a line graph with error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_values: The y-axis values.
        y_mean: The mean y-axis values.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
        confidence_level: Confidence level for the interval.
    """
    errors = calculate_errors(y_values, y_mean, confidence_level=confidence_level)

    for index, (start, end) in enumerate(segments):
        axis.errorbar(
            x_values[start:end],
            y_mean[start:end],
            errors[:, start:end],
            label=label if index == 0 else None,
            color=color,
            capsize=3,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_without_error_bars(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_mean: Real[Array, " values"],
    segments: list[tuple[int, int]],
    label: str,
    color: str,
) -> None:
    """Plot a line graph without error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_mean: The mean y-axis values.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
    """
    for index, (start, end) in enumerate(segments):
        axis.plot(
            x_values[start:end],
            y_mean[start:end],
            label=label if index == 0 else None,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_data(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_values: Real[Array, " trial_count values"],
    label: str,
    color: str,
    *,
    ci_mode: Literal["omit", "subjectwise"] = "subjectwise",
    confidence_level: float = 0.95,
) -> Axes:
    """Plot mean curves with optional bootstrap error bars.

    Args:
        axis: the axis to plot on; if None, a new figure is created.
        x_values: Vector of x values to plot.
        y_values: vector or matrix of y values to plot.
        label: name for the plotted data.
        color: color for the plotted data.
        ci_mode: Whether to show subjectwise error bars or omit them.
        confidence_level: Confidence level for the interval.
    """
    if y_values.ndim == 1:
        y_values = y_values[jnp.newaxis, :]
    y_mean = jnp.nanmean(y_values, axis=0)
    segments = segment_by_nan(y_mean)

    if len(y_values) > 1 and ci_mode != "omit":
        plot_with_error_bars(
            axis,
            x_values,
            y_values,
            y_mean,
            segments,
            label,
            color,
            confidence_level=confidence_level,
        )
    else:
        plot_without_error_bars(axis, x_values, y_mean, segments, label, color)

    x = jnp.asarray(x_values)
    x = x[jnp.isfinite(x)]
    if x.size and bool(jnp.allclose(x, jnp.round(x), atol=1e-9)):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    return axis


def _subject_ratio_curves(
    numerator_by_trial: np.ndarray,
    denominator_by_trial: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Return per-subject ratio curves.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.

    Returns:
        Array of ratio curves with one row per subject.
    """
    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size == 0:
        raise ValueError("No subjects available after applying trial_mask.")

    subject_curves: list[np.ndarray] = []
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        numerator_sum = numerator_by_trial[subject_mask].sum(axis=0)
        denominator_sum = denominator_by_trial[subject_mask].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            subject_curves.append(numerator_sum / denominator_sum)
    return np.vstack(subject_curves)


def _bootstrap_subjectwise_ratio(
    subject_ratio_curves: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from subjectwise bootstrap sampling.

    Args:
        subject_ratio_curves: Ratio curves per subject.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    subject_count = subject_ratio_curves.shape[0]
    sample_indices = rng.integers(
        0, subject_count, size=(n_resamples, subject_count)
    )
    sampled_curves = subject_ratio_curves[sample_indices]
    mean_curves = np.nanmean(sampled_curves, axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = np.nanpercentile(mean_curves, 100 * alpha, axis=0)
    ci_high = np.nanpercentile(mean_curves, 100 * (1 - alpha), axis=0)
    return ci_low, ci_high


def _bootstrap_trialwise_ratio(
    numerator_by_trial: np.ndarray,
    denominator_by_trial: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from trialwise bootstrap sampling.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    trial_count = numerator_by_trial.shape[0]
    sample_indices = rng.integers(0, trial_count, size=(n_resamples, trial_count))
    numerator_samples = numerator_by_trial[sample_indices].sum(axis=1)
    denominator_samples = denominator_by_trial[sample_indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_samples = numerator_samples / denominator_samples
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = np.nanpercentile(ratio_samples, 100 * alpha, axis=0)
    ci_high = np.nanpercentile(ratio_samples, 100 * (1 - alpha), axis=0)
    return ci_low, ci_high


def _bootstrap_hierarchical_ratio(
    numerator_by_trial: np.ndarray,
    denominator_by_trial: np.ndarray,
    subject_ids: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from hierarchical bootstrap sampling.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    unique_subjects = np.unique(subject_ids)
    indices_by_subject: dict[int, np.ndarray] = {}
    for subject in unique_subjects:
        indices_by_subject[int(subject)] = np.nonzero(subject_ids == subject)[0]

    boot_curves: list[np.ndarray] = []
    for _ in range(n_resamples):
        sampled_subjects = rng.choice(
            unique_subjects, size=unique_subjects.size, replace=True
        )
        numerator_sum = np.zeros(numerator_by_trial.shape[1], dtype=float)
        denominator_sum = np.zeros(denominator_by_trial.shape[1], dtype=float)
        for subject in sampled_subjects:
            subject_indices = indices_by_subject[int(subject)]
            resampled_indices = rng.choice(
                subject_indices, size=subject_indices.size, replace=True
            )
            numerator_sum += numerator_by_trial[resampled_indices].sum(axis=0)
            denominator_sum += denominator_by_trial[resampled_indices].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_curves.append(numerator_sum / denominator_sum)

    boot_curves_array = np.vstack(boot_curves)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = np.nanpercentile(boot_curves_array, 100 * alpha, axis=0)
    ci_high = np.nanpercentile(boot_curves_array, 100 * (1 - alpha), axis=0)
    return ci_low, ci_high


def _plot_with_confidence_interval(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    segments: list[tuple[int, int]],
    label: str,
    color: str,
) -> None:
    """Plot a line graph with explicit confidence bounds.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_mean: The mean y-axis values.
        ci_low: Lower confidence bounds.
        ci_high: Upper confidence bounds.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
    """
    errors = np.vstack((y_mean - ci_low, ci_high - y_mean))
    for index, (start, end) in enumerate(segments):
        axis.errorbar(
            x_values[start:end],
            y_mean[start:end],
            errors[:, start:end],
            label=label if index == 0 else None,
            color=color,
            capsize=3,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_ratio_data(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    numerator_by_trial: Real[Array, " trial_count values"],
    denominator_by_trial: Real[Array, " trial_count values"],
    subject_ids: Integer[Array, " trial_count"],
    trial_mask: Bool[Array, " trial_count"],
    label: str,
    color: str,
    *,
    ci_mode: BootstrapMode = "hierarchical",
    n_resamples: int = 100,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Axes:
    """Plot ratio curves with configurable bootstrap confidence intervals.

    Args:
        axis: Axis to plot on.
        x_values: X-axis values.
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
        trial_mask: Boolean mask selecting trials to include.
        label: Legend label for the line.
        color: Line and error bar color.
        ci_mode: Bootstrap procedure to use for confidence intervals.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: Optional NumPy random generator.

    Notes:
        - The mean curve is the average of per-subject ratios computed after
          applying ``trial_mask``.
        - When ``ci_mode`` is not ``"omit"`` and only one subject is present,
          the confidence interval falls back to trialwise sampling.
    """
    trial_mask_np = np.asarray(trial_mask, dtype=bool).reshape(-1)
    if not np.any(trial_mask_np):
        raise ValueError("No trials selected by trial_mask.")

    numerator = np.asarray(numerator_by_trial)[trial_mask_np]
    denominator = np.asarray(denominator_by_trial)[trial_mask_np]
    subject_ids_np = np.asarray(subject_ids).reshape(-1)[trial_mask_np]
    subject_ratio_curves = _subject_ratio_curves(
        numerator_by_trial=numerator,
        denominator_by_trial=denominator,
        subject_ids=subject_ids_np,
    )
    y_mean = np.nanmean(subject_ratio_curves, axis=0)
    segments = segment_by_nan(jnp.asarray(y_mean))

    if rng is None:
        rng = np.random.default_rng()

    subject_count = subject_ratio_curves.shape[0]
    effective_mode = ci_mode
    if ci_mode != "omit" and subject_count <= 1:
        effective_mode = "trialwise"

    if effective_mode == "omit" or n_resamples < 1:
        plot_without_error_bars(axis, x_values, y_mean, segments, label, color)
    else:
        if effective_mode == "subjectwise":
            ci_low, ci_high = _bootstrap_subjectwise_ratio(
                subject_ratio_curves=subject_ratio_curves,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
        elif effective_mode == "trialwise":
            ci_low, ci_high = _bootstrap_trialwise_ratio(
                numerator_by_trial=numerator,
                denominator_by_trial=denominator,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
        elif effective_mode == "hierarchical":
            ci_low, ci_high = _bootstrap_hierarchical_ratio(
                numerator_by_trial=numerator,
                denominator_by_trial=denominator,
                subject_ids=subject_ids_np,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported ci_mode: {ci_mode}")
        _plot_with_confidence_interval(
            axis,
            x_values,
            np.asarray(y_mean),
            np.asarray(ci_low),
            np.asarray(ci_high),
            segments,
            label,
            color,
        )

    x = jnp.asarray(x_values)
    x = x[jnp.isfinite(x)]
    if x.size and bool(jnp.allclose(x, jnp.round(x), atol=1e-9)):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    return axis


def set_plot_labels(
    axis: Axes, xlabel: str, ylabel: str, contrast_name: Optional[str] = None
) -> Axes:
    """Returns an axis with modified labels and optional legend settings for a plot.

    Args:
        axis: The axis to modify.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        contrast_name: Name of the contrast for the legend, if applicable.
    """
    # Set x and y labels
    axis.set(xlabel=xlabel, ylabel=ylabel)

    # If a contrast name is provided, set it as the legend's title
    if contrast_name:
        axis.legend(title=contrast_name)

    axis.tick_params(labelsize=14)
    axis.set_xlabel(axis.get_xlabel(), fontsize=16)
    axis.set_ylabel(axis.get_ylabel(), fontsize=16)
    for loc in ("top", "right"):
        axis.spines[loc].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    return axis
