from typing import Optional

import matplotlib.pyplot as plt
from jax import numpy as jnp
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import bootstrap

from jaxcmr.typing import Array, Real

__all__ = [
    "init_plot",
    "plot_data",
    "calculate_errors",
    "plot_with_error_bars",
    "plot_without_error_bars",
    "set_plot_labels",
]


def segment_by_nan(vector: jnp.ndarray) -> list[tuple[int, int]]:
    "Returns list of tuples segmenting the vector by its NaN values."
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
    """Return initialized plotting axis

    Args:
        axis: An existing axis to use for the plot.
    """
    if axis is None:
        plt.figure()
        axis = plt.gca()
    return axis


def calculate_errors(y_values: jnp.ndarray, y_mean: jnp.ndarray) -> jnp.ndarray:
    """Returns an array of errors calculated from the given y values and y mean, using bootstrapped confidence intervals.

    Args:
        y_values: Array of y values.
        y_mean: Array of y mean values.
    """
    errors = jnp.zeros((2, len(y_mean)))
    bootstrapped_confidence_intervals = bootstrap(
        (y_values,), jnp.nanmean, confidence_level=0.95
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
):
    """Returns a plot of a line graph with error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_values: The y-axis values.
        y_mean: The mean y-axis values.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
    """
    errors = calculate_errors(y_values, y_mean)

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
):
    """Returns a plot of a line graph without error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_values: The y-axis values.
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
) -> Axes:
    """Returns axis plotting data as line segment(s) on the provided axis.

    Args:
        axis: the axis to plot on; if None, a new figure is created.
        x_values: Vector of x values to plot.
        y_values: vector or matrix of y values to plot.
        label: name for the plotted data.
        color: color for the plotted data.
    """
    if y_values.ndim == 1:
        y_values = y_values[jnp.newaxis, :]
    y_mean = jnp.nanmean(y_values, axis=0)
    segments = segment_by_nan(y_mean)

    if len(y_values) > 1:
        plot_with_error_bars(axis, x_values, y_values, y_mean, segments, label, color)
    else:
        plot_without_error_bars(axis, x_values, y_mean, segments, label, color)

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
