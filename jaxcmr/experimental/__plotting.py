"""Experimental plotting with hierarchical bootstrap CIs for ratio analyses.

Extends ``jaxcmr.plotting`` with ``compute_ratio_ci`` and ``plot_ratio_data``,
which compute bootstrapped confidence intervals for ratio curves (e.g.,
conditional recall probability = recalled / presented).  Unlike the simple
bootstrap in ``plotting.py`` (which treats all trials as exchangeable), the
hierarchical bootstrap here resamples *subjects* first, correctly accounting
for subject-level clustering.  This matters for analyses that compare recall
rates across conditions (emotional vs. neutral, by serial position, etc.)
where ignoring the nested structure would underestimate CI width.

The remaining functions (``init_plot``, ``plot_data``, etc.) duplicate
``jaxcmr.plotting`` and exist only to keep this module self-contained.

"""

from collections.abc import Sequence
from typing import Callable, Optional, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import bootstrap

from jaxcmr.typing import Array, Bool, Integer, Real, RecallDataset

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "compute_ratio_ci",
    "init_plot",
    "prepare_plot_inputs",
    "plot_data",
    "plot_ratio_data",
    "calculate_errors",
    "plot_with_error_bars",
    "plot_without_error_bars",
    "set_plot_labels",
]

DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000
DEFAULT_CI_TRANSFORM: str = "arcsin_sqrt"

T = TypeVar("T")


def _normalize_to_list(value: Sequence[T] | T) -> list[T]:
    """Returns value as a list.

    Args:
        value: Value to normalize into a list.
    """
    if isinstance(value, dict):
        return [cast(T, value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(cast(Sequence[T], value))
    return [cast(T, value)]


def prepare_plot_inputs(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]],
    axis: Optional[Axes],
) -> tuple[
    Axes,
    list[RecallDataset],
    list[Bool[Array, " trial_count"]],
    list[str],
]:
    """Returns normalized plot inputs for analysis plotters.

    Args:
        datasets: Datasets to plot.
        trial_masks: Masks selecting trials for each dataset.
        color_cycle: Colors for plotting each dataset.
        axis: Existing axis to plot on.

    Returns:
        (axis, datasets, trial_masks, color_cycle): Normalized plot inputs.
    """
    axis = init_plot(axis)
    datasets_list = _normalize_to_list(datasets)
    trial_masks_list = _normalize_to_list(trial_masks)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]
    return axis, datasets_list, trial_masks_list, color_cycle


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
    ci_mode: str = "subjectwise",
    confidence_level: float = 0.95,
) -> Axes:
    """Plot mean curves with optional bootstrap error bars.

    Args:
        axis: the axis to plot on
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
    subject_ids,
) -> np.ndarray:
    """Returns array of ratio curves with one row per subject.

    Each curve is computed as the sum of per-trial numerators divided by the
    sum of per-trial denominators for that subject.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
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
    ci_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from subjectwise bootstrap sampling.

    Args:
        subject_ratio_curves: Ratio curves per subject.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.
        ci_transform: CI transform to apply before percentile bounds.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    subject_count = subject_ratio_curves.shape[0]
    sample_indices = rng.integers(0, subject_count, size=(n_resamples, subject_count))
    sampled_curves = subject_ratio_curves[sample_indices]
    mean_curves = np.nanmean(sampled_curves, axis=1)
    return _percentile_ci_bounds(
        mean_curves,
        confidence_level=confidence_level,
        ci_transform=ci_transform,
    )


def _bootstrap_trialwise_ratio(
    numerator_by_trial: np.ndarray,
    denominator_by_trial: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
    ci_transform: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from trialwise bootstrap sampling.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.
        ci_transform: CI transform to apply before percentile bounds.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    trial_count = numerator_by_trial.shape[0]
    sample_indices = rng.integers(0, trial_count, size=(n_resamples, trial_count))
    numerator_samples = numerator_by_trial[sample_indices].sum(axis=1)
    denominator_samples = denominator_by_trial[sample_indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_samples = numerator_samples / denominator_samples
    if transform is not None:
        ratio_samples = transform(ratio_samples)
    return _percentile_ci_bounds(
        ratio_samples,
        confidence_level=confidence_level,
        ci_transform=ci_transform,
    )


def _bootstrap_hierarchical_ratio(
    numerator_by_trial: np.ndarray,
    denominator_by_trial: np.ndarray,
    subject_ids: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
    ci_transform: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from hierarchical bootstrap sampling.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: NumPy random generator.
        ci_transform: CI transform to apply before percentile bounds.

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
            boot_curve = numerator_sum / denominator_sum
        if transform is not None:
            boot_curve = transform(boot_curve)
        boot_curves.append(boot_curve)

    boot_curves_array = np.vstack(boot_curves)
    return _percentile_ci_bounds(
        boot_curves_array,
        confidence_level=confidence_level,
        ci_transform=ci_transform,
    )


def _arcsin_sqrt_forward(proportions: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Return arcsin-sqrt transformed proportions.

    Args:
        proportions: Proportions in [0, 1]. NaNs are allowed.
        tol: Allowed numerical tolerance outside [0, 1] before raising.
    """
    finite = np.isfinite(proportions)
    if np.any(proportions[finite] < -tol) or np.any(proportions[finite] > 1.0 + tol):
        raise ValueError("arcsin-sqrt CI requires proportions in [0, 1].")
    clipped = np.clip(proportions, 0.0, 1.0)
    return np.arcsin(np.sqrt(clipped))


def _arcsin_sqrt_inverse(angles: np.ndarray) -> np.ndarray:
    """Return inverse arcsin-sqrt transform."""
    return np.sin(angles) ** 2


def _percentile_ci_bounds(
    bootstrap_statistics: np.ndarray,
    *,
    confidence_level: float,
    ci_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence bounds from bootstrap statistics.

    Args:
        bootstrap_statistics: Bootstrap statistic array with shape
            [n_resamples, n_points].
        confidence_level: Confidence level for the bounds.
        ci_transform: Transform applied before percentile bounds.
            Use ``\"none\"`` for raw percentile bounds or ``\"arcsin_sqrt\"`` for
            arcsin-sqrt transform, percentile bounds, and inverse transform.

    Returns:
        Tuple of (ci_low, ci_high) arrays.
    """
    alpha = (1.0 - confidence_level) / 2.0
    low_pct = 100 * alpha
    high_pct = 100 * (1 - alpha)

    normalized = ci_transform.replace("-", "_").lower()
    if normalized in ("none", "identity", "raw"):
        ci_low = np.nanpercentile(bootstrap_statistics, low_pct, axis=0)
        ci_high = np.nanpercentile(bootstrap_statistics, high_pct, axis=0)
        return ci_low, ci_high
    if normalized in ("arcsin_sqrt", "asin_sqrt", "angular"):
        transformed = _arcsin_sqrt_forward(bootstrap_statistics)
        ci_low_t = np.nanpercentile(transformed, low_pct, axis=0)
        ci_high_t = np.nanpercentile(transformed, high_pct, axis=0)
        return _arcsin_sqrt_inverse(ci_low_t), _arcsin_sqrt_inverse(ci_high_t)

    raise ValueError(f"Unsupported ci_transform: {ci_transform}")


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


def compute_ratio_ci(
    numerator_by_trial: Real[Array, " trial_count values"],
    denominator_by_trial: Real[Array, " trial_count values"],
    subject_ids: Integer[Array, " trial_count"],
    *,
    ci_mode: str = "hierarchical",
    ci_transform: str = DEFAULT_CI_TRANSFORM,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute mean ratio curve and bootstrap confidence bounds.

    Args:
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
        ci_mode: Bootstrap procedure for confidence intervals.
        ci_transform: Transform applied to bootstrap statistics before computing
            percentile bounds.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: Optional NumPy random generator.
        transform: Optional transform applied to ratio curves before averaging.
            If it changes the number of values, callers must align x-values.

    Returns:
        (y_mean, ci_low, ci_high): Mean ratio curve and confidence bounds.
            Bounds are ``None`` when ``ci_mode`` is ``"omit"`` or
            ``n_resamples < 1``.

    Notes:
        - When ``ci_mode`` is not ``"omit"`` and only one subject is present,
          the confidence interval falls back to trialwise sampling.
        - ``transform`` is applied to per-subject ratios and bootstrap samples.
        - ``ci_transform`` is applied to bootstrap statistics before percentile
          bounds are computed.
        - ``transform`` must handle NaN inputs.
    """
    num = np.array(numerator_by_trial)
    denom = np.array(denominator_by_trial)
    subject_ids_np = np.array(subject_ids)
    subject_ratio_curves = _subject_ratio_curves(num, denom, subject_ids_np)
    if transform is not None:
        subject_ratio_curves = transform(subject_ratio_curves)
    y_mean = np.nanmean(subject_ratio_curves, axis=0)

    if ci_mode == "omit" or n_resamples < 1:
        return y_mean, None, None

    if rng is None:
        rng = np.random.default_rng()

    subject_count = subject_ratio_curves.shape[0]
    effective_mode = ci_mode
    if ci_mode != "omit" and subject_count <= 1:
        effective_mode = "trialwise"

    if effective_mode == "subjectwise":
        ci_low, ci_high = _bootstrap_subjectwise_ratio(
            subject_ratio_curves=subject_ratio_curves,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            rng=rng,
            ci_transform=ci_transform,
        )
    elif effective_mode == "trialwise":
        ci_low, ci_high = _bootstrap_trialwise_ratio(
            num,
            denom,
            n_resamples,
            confidence_level,
            rng=rng,
            ci_transform=ci_transform,
            transform=transform,
        )
    elif effective_mode == "hierarchical":
        ci_low, ci_high = _bootstrap_hierarchical_ratio(
            num,
            denom,
            subject_ids_np,
            n_resamples,
            confidence_level,
            rng=rng,
            ci_transform=ci_transform,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported ci_mode: {ci_mode}")

    return y_mean, ci_low, ci_high


def plot_ratio_data(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    numerator_by_trial: Real[Array, " trial_count values"],
    denominator_by_trial: Real[Array, " trial_count values"],
    subject_ids: Integer[Array, " trial_count"],
    label: str,
    color: str,
    *,
    ci_mode: str = "hierarchical",
    ci_transform: str = DEFAULT_CI_TRANSFORM,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Axes:
    """Plot ratio curves with configurable bootstrap confidence intervals.

    Args:
        axis: Axis to plot on.
        x_values: X-axis values.
        numerator_by_trial: Numerator sums per trial.
        denominator_by_trial: Denominator sums per trial.
        subject_ids: Subject id per trial.
        label: Legend label for the line.
        color: Line and error bar color.
        ci_mode: Bootstrap procedure to use for confidence intervals.
        ci_transform: Transform applied to bootstrap statistics before computing
            percentile bounds.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for the bounds.
        rng: Optional NumPy random generator.
        transform: Optional transform applied to ratio curves before averaging.
            If it changes the number of values, ``x_values`` must match.

    Notes:
        - The mean curve is the average of per-subject ratios
        - When ``ci_mode`` is not ``"omit"`` and only one subject is present,
          the confidence interval falls back to trialwise sampling.
        - ``transform`` is applied to per-subject ratios and bootstrap samples.
        - ``ci_transform`` is applied to bootstrap statistics before percentile
          bounds are computed.
        - ``transform`` must handle NaN inputs.
    """
    y_mean, ci_low, ci_high = compute_ratio_ci(
        numerator_by_trial,
        denominator_by_trial,
        subject_ids,
        ci_mode=ci_mode,
        ci_transform=ci_transform,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
        transform=transform,
    )
    y_mean_array = jnp.asarray(y_mean)
    segments = segment_by_nan(y_mean_array)

    if ci_low is None or ci_high is None:
        plot_without_error_bars(axis, x_values, y_mean_array, segments, label, color)
    else:
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
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)

    if contrast_name:
        axis.legend(title=contrast_name)

    axis.tick_params(labelsize=14)
    for loc in ("top", "right"):
        axis.spines[loc].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    return axis
