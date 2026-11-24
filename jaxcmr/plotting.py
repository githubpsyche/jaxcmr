"""Draw behavioral summaries and arrange comparisons across datasets and conditions.

If you have a recall dataset, start with the plotter for the measure you
want, such as ``jaxcmr.analyses.spc.plot_spc``. It selects trials, calculates
a serial position curve for each participant, and passes those curves to
``plot_mean_curve``. If you already have participant scores, you can call
``plot_mean_curve`` directly. Each row is an observation and each column a
position or bin. The mean and optional bootstrap interval therefore give
each participant with a finite score equal weight in that column.

When writing an analysis plotter, use ``prepare_plot_inputs`` to obtain axes,
trial selections and colors. Several mean curves can share those axes.
``format_analysis_axes`` labels them after the curves are drawn. For a
matrix of values, ``matrix_heatmap`` displays the cells on a common color
scale.

``comparison_panels`` arranges datasets and trial selections before an
analysis runs. ``iter_condition_curves`` works with participant scores that
have already been calculated, separating them into curves for individual
conditions. To run several configured analyses, ``iter_analysis_figures``
calls their plotters and yields each figure for display or saving.
"""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator
from typing_extensions import TypedDict

from jaxcmr.analyses.statistics import _MeanInterval, mean_interval
from jaxcmr.dataset import generate_trial_mask
from jaxcmr.helpers import import_from_string
from jaxcmr.typing import Array, Bool, Float, Real, RecallDataset

__all__ = [
    "add_analysis_legend",
    "bootstrap_mean_errors",
    "comparison_panels",
    "format_analysis_axes",
    "init_plot",
    "iter_analysis_figures",
    "iter_condition_curves",
    "matrix_heatmap",
    "plot_mean_curve",
    "plot_mean_intervals",
    "prepare_plot_inputs",
    "save_figure",
]


class _ComparisonPanel(TypedDict, extra_items=Any):
    """Selected datasets, masks and labels for one comparison panel."""

    datasets: list[RecallDataset]
    trial_masks: list[Bool[Array, " trials"]]
    labels: list[str]
    title: str


def init_plot(axis: Axes | None = None) -> Axes:
    """Return the supplied axes, or create axes in a new figure.

    Parameters
    ----------
    axis : Axes, optional
        Existing axes to draw on. Omit to create a figure using Matplotlib's
        current figure size and automatic spacing.

    Returns
    -------
    Axes
        The supplied or newly created axes, without additional styling.
    """
    if axis is not None:
        return axis
    _, new_axis = plt.subplots(tight_layout=True)
    return new_axis


def _normalize_to_list[T](value: Sequence[T] | T) -> list[T]:
    """Wrap a single dataset or array; copy an existing sequence into a list."""
    if isinstance(value, dict):
        return [cast(T, value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(cast(Sequence[T], value))
    return [cast(T, value)]


def prepare_plot_inputs(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None,
    axis: Axes | None,
) -> tuple[
    Axes,
    list[RecallDataset],
    list[Bool[Array, " trial_count"]],
    list[str],
]:
    """Prepare collections and plotting defaults for an analysis plotter.

    Parameters
    ----------
    datasets : RecallDataset or sequence of RecallDataset
        One dataset or an ordered collection; dataset arrays are not copied.
    trial_masks : Boolean array or sequence of Boolean arrays
        One aligned trial selection per dataset, in the same order.
    color_cycle : list of str or None
        Nonempty list of colors, or None to use Matplotlib's current color cycle.
    axis : Axes or None
        Existing axes, or None to create axes in a new figure.

    Returns
    -------
    tuple
        Axes, dataset list, mask list and color list. Selection evaluation and
        score calculation remain the caller's responsibility.
    """
    datasets_list = _normalize_to_list(datasets)
    trial_masks_list = _normalize_to_list(trial_masks)
    axis = init_plot(axis)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]
    return axis, datasets_list, trial_masks_list, color_cycle


def bootstrap_mean_errors(
    y_values: Real[Array | np.ndarray, " observations bins"],
    y_mean: Real[Array | np.ndarray, " bins"],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Float[Array, "2 bins"]:
    """Return finite-observation bootstrap error distances around a mean curve.

    Parameters
    ----------
    y_values : array of shape (observations, bins)
        Independent observations, usually participant scores. Each bin uses its
        finite observations with equal weight.
    y_mean : array of shape (bins,)
        Finite-observation column means used as the centers.
    confidence_level : float, optional
        BCa coverage, default 0.95.
    n_resamples : int, optional
        Bootstrap sample count, default 9999.
    bootstrap_seed : int, optional
        Reproducible seed. One generator supplies successive column resamples.

    Returns
    -------
    array of shape (2, bins)
        Lower and upper error distances. Fewer than two finite observations
        gives missing limits; constant samples give zero-length intervals.
    """
    generator = np.random.default_rng(bootstrap_seed)
    errors = []
    for values, mean in zip(np.asarray(y_values).T, np.asarray(y_mean), strict=True):
        interval = mean_interval(
            values,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=generator,
        )
        errors.append((mean - interval["low"], interval["high"] - mean))
    return jnp.asarray(errors).T


def plot_mean_curve(
    axis: Axes,
    x_values: Real[Array | np.ndarray, " bins"],
    y_values: Real[Array | np.ndarray, " bins"]
    | Real[Array | np.ndarray, " observations bins"],
    label: str,
    color: str,
    *,
    draw_ci: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Draw the mean across observation rows, optionally with bootstrap intervals.

    Parameters
    ----------
    axis : Axes
        Existing axes to draw on; this operation does not create a figure.
    x_values : array of shape (bins,)
        Positions of the plotted bins. Integer-valued positions use integer ticks.
    y_values : array of shape (bins,) or (observations, bins)
        Values by bin, optionally with multiple independent observation rows.
        Each finite row contributes equally within a bin. Analysis plotters
        normally pass participant scores; this function does not group trials.
        A vector is treated as one observation and receives no error bars.
    label : str
        Legend label for the curve, including curves interrupted by missing bins.
    color : str
        Color of the line, markers and error bars.
    draw_ci : bool, optional
        Draw bootstrap intervals when there is more than one observation row.
    confidence_level : float, optional
        Coverage passed to bootstrap_mean_errors.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        The supplied axes with the curve added. Bins with a NaN mean break the
        line; Matplotlib omits their markers and error bars.
    """
    if y_values.ndim == 1:
        y_values = y_values[jnp.newaxis, :]
    values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    y_mean = np.divide(
        np.where(finite, values, 0).sum(axis=0),
        counts,
        out=np.full(values.shape[1], np.nan),
        where=counts > 0,
    )

    if len(y_values) > 1 and draw_ci:
        errors = bootstrap_mean_errors(
            y_values,
            y_mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        axis.errorbar(
            x_values,
            y_mean,
            errors,
            label=label,
            color=color,
            capsize=3,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )
    else:
        axis.plot(
            x_values,
            y_mean,
            label=label,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )

    finite_positions = jnp.asarray(x_values)
    finite_positions = finite_positions[jnp.isfinite(finite_positions)]
    if finite_positions.size and bool(
        jnp.allclose(finite_positions, jnp.round(finite_positions), atol=1e-9)
    ):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    return axis


def add_analysis_legend(axis: Axes, title: str | None = None) -> Legend:
    """Place the legend to the right of the axes, clear of curves and error bars.

    Parameters
    ----------
    axis : Axes
        Axes containing the labeled artists. For an existing figure, use
        tight or constrained layout to reserve space for the legend.
    title : str, optional
        Heading above the legend entries.

    Returns
    -------
    Legend
        The legend attached to the supplied axes.
    """
    return axis.legend(
        title=title, loc="upper left", bbox_to_anchor=(1, 1), frameon=False
    )


def format_analysis_axes(
    axis: Axes, xlabel: str, ylabel: str, contrast_name: str | None = None
) -> Axes:
    """Apply labels, an optional legend and the shared analysis-plot style.

    Parameters
    ----------
    axis : Axes
        Existing axes to format.
    xlabel, ylabel : str
        Labels for the horizontal and vertical axes.
    contrast_name : str, optional
        Legend title. A nonempty value adds the plotted labels in the right margin.

    Returns
    -------
    Axes
        The supplied axes with label and tick sizes set, top and right spines
        hidden, and horizontal grid lines drawn behind the data.
    """
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)

    if contrast_name:
        add_analysis_legend(axis, contrast_name)

    axis.tick_params(labelsize=14)
    for loc in ("top", "right"):
        axis.spines[loc].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#e8e8e8", linewidth=0.8)

    return axis


def plot_mean_intervals(
    axis: Axes,
    intervals: Sequence[_MeanInterval],
    labels: Sequence[str],
    *,
    ylabel: str,
    color_cycle: Sequence[str] | None = None,
    contrast_name: str | None = None,
) -> Axes:
    """Draw one point and its precomputed confidence interval for each group.

    Parameters
    ----------
    axis : Axes
        Axes on which to draw the points.
    intervals : sequence of mean_interval results
        One summary per group. The point uses ``mean`` and the error bar
        extends from ``low`` to ``high``. Missing limits leave the point alone.
    labels : sequence of str
        Group names in display order along the horizontal axis.
    ylabel : str
        Name of the measured quantity on the vertical axis.
    color_cycle : sequence of str, optional
        Marker colors in group order.
    contrast_name : str, optional
        Title for the group legend.

    Returns
    -------
    Axes
        The axes containing the group means and intervals.
    """
    if color_cycle is None:
        color_cycle = [entry["color"] for entry in rcParams["axes.prop_cycle"]]
    for index, (interval, label) in enumerate(zip(intervals, labels, strict=True)):
        mean, low, high = interval["mean"], interval["low"], interval["high"]
        axis.plot(index, mean, "o", color=color_cycle[index % len(color_cycle)],
                  markersize=6, label=label, zorder=3)
        axis.vlines(index, low, high, color="black")
        axis.plot([index, index], [low, high], "_", color="black", markersize=10)
    axis.set_xticks(np.arange(len(labels)), labels, fontsize=14)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    return format_analysis_axes(axis, "", ylabel, contrast_name)


def matrix_heatmap(
    matrix: Real[Array | np.ndarray, " values"]
    | Real[Array | np.ndarray, " rows columns"],
    figsize: tuple[float, float] = (15, 15),
    axis: Axes | None = None,
    label_font_size: float = 32,
    annot_font_size: float = 14,
    print_threshold: float = 0.005,
    title: str = "",
    xlabel: str = "Column Index",
    ylabel: str = "Row Index",
    *,
    colorbar: bool = True,
    norm: Normalize | None = None,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """Draw a vector or matrix as a value-annotated heatmap.

    Parameters
    ----------
    matrix : array of shape (values,) or (rows, columns)
        Values to display. A vector is drawn as one row; rows run from top to
        bottom. Nonfinite cells are blank. The input is not modified.
    figsize : tuple of float, optional
        Width and height in inches when creating a figure. Ignored with axis.
    axis : Axes, optional
        Existing axes to draw on. Omit to create a figure.
    label_font_size : float, optional
        Font size for the title and axis labels.
    annot_font_size : float, optional
        Font size for numbers drawn inside cells.
    print_threshold : float, optional
        Annotate finite values whose absolute magnitude is at least this value.
        Annotations use two decimal places; this threshold does not affect color.
    title : str, optional
        Title of the heatmap.
    xlabel : str, optional
        Horizontal axis label. Supply the scientific meaning of the columns.
    ylabel : str, optional
        Vertical axis label. Supply the scientific meaning of the rows.
    colorbar : bool, optional
        Add a colorbar to the figure.
    norm : matplotlib.colors.Normalize, optional
        Mapping from values to colors. Supply a common normalization with fixed
        limits to compare several matrices on the same scale. By default each
        heatmap scales to its own finite values. The palette is magma.
    row_labels, column_labels : sequence of str, optional
        One label per displayed row or column. Defaults to zero-based indices.
        Ticks are thinned to fit the available space; supplied labels stay
        attached to their corresponding cells.

    Returns
    -------
    tuple of Figure or SubFigure and Axes
        The containing figure or subfigure and heatmap axes. Drawing does not
        show, save or close them.

    Examples
    --------
    Set fixed color limits that can be reused for other heatmaps:

    >>> from matplotlib.colors import Normalize
    >>> scale = Normalize(vmin=0, vmax=1)
    >>> figure, axis = matrix_heatmap(np.array([[0.2, 0.8]]), norm=scale)
    """
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize)
    else:
        figure = axis.figure

    mesh = axis.pcolormesh(
        np.ma.masked_invalid(matrix),
        cmap="magma",
        norm=norm,
        linewidths=0.5,
        edgecolors="white",
        shading="flat",
    )
    row_count, column_count = matrix.shape
    axis.set_xlim(0, column_count)
    axis.set_ylim(row_count, 0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if colorbar:
        figure.colorbar(mesh, ax=axis)

    # Thin ticks while retaining the caller's row and column identities.
    for dimension, count, labels in (
        (axis.xaxis, column_count, column_labels),
        (axis.yaxis, row_count, row_labels),
    ):
        tick_capacity = max(1, dimension.get_tick_space())
        tick_step = max(1, int(np.ceil(count / tick_capacity)))
        indices = np.arange(0, count, tick_step)
        tick_labels = [
            str(index) if labels is None else labels[int(index)] for index in indices
        ]
        dimension.set_ticks(indices + 0.5, labels=tick_labels)

    for (row, column), value in np.ndenumerate(matrix):
        if not np.isfinite(value) or -print_threshold < value < print_threshold:
            continue
        colour = mesh.cmap(mesh.norm(value))
        brightness = np.dot(colour[:3], (0.299, 0.587, 0.114))
        axis.text(
            column + 0.5,
            row + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=annot_font_size,
            color="black" if brightness > 0.5 else "white",
        )

    axis.set_xlabel(xlabel, fontsize=label_font_size)
    axis.set_ylabel(ylabel, fontsize=label_font_size)
    axis.set_title(title, fontsize=label_font_size)

    return figure, axis


def iter_condition_curves(
    subject_values: Sequence[Real[Array, " bins"] | Real[Array, " conditions bins"]],
    condition_count: int,
    dataset_index: int,
    labels: Sequence[str],
    color_cycle: Sequence[str],
) -> Iterator[tuple[Real[Array, " subjects bins"], str, str]]:
    """Arrange participant results into curves with matching labels and colors.

    Parameters
    ----------
    subject_values : sequence of arrays
        One array per participant: bins for an unconditioned analysis, or
        conditions by bins for a conditioned analysis.
    condition_count : int
        Number of conditions. Zero denotes one unconditioned curve.
    dataset_index : int
        Position of this dataset in the caller's comparison.
    labels : sequence of str
        Labels selected according to condition_count and the label count.
        The selection rules are listed in Notes below.
    color_cycle : sequence of str
        Nonempty colors, cycled in dataset/condition order.

    Yields
    ------
    tuple
        Participant-by-bin values, label and color for each condition in order,
        or one tuple for an unconditioned analysis. No averaging or drawing occurs.

    Notes
    -----
    The label rules are applied in this order:

    | Condition count | Label count | Labels describe |
    | --- | --- | --- |
    | Zero | Any | Datasets, indexed by dataset_index. |
    | Positive | Greater than condition_count | Dataset/condition pairs, with conditions varying fastest. |
    | Greater than one | At most condition_count | Shared conditions, indexed by condition. |
    | One | At most condition_count | Datasets, indexed by dataset_index. |

    Supply enough labels for the selected rule and nonempty colors. Colors cycle
    in dataset order for unconditioned curves, or dataset/condition order for
    conditioned curves. Participant order is preserved within every curve.
    """
    if not condition_count:
        yield (
            jnp.vstack(subject_values),
            labels[dataset_index],
            color_cycle[dataset_index % len(color_cycle)],
        )
        return
    for condition_index in range(condition_count):
        values = jnp.vstack([subject[condition_index] for subject in subject_values])
        color_index = dataset_index * condition_count + condition_index
        label = (
            labels[color_index]
            if len(labels) > condition_count
            else labels[condition_index]
            if condition_count > 1
            else labels[dataset_index]
        )
        yield values, label, color_cycle[color_index % len(color_cycle)]


def comparison_panels(
    datasets: Sequence[RecallDataset],
    masks_by_dataset: Sequence[Sequence[Bool[Array, " trials"] | None]],
    dataset_labels: Sequence[str],
    condition_labels: Sequence[str],
    *,
    mode: Literal["conditions", "datasets"],
    selection_label: str = "Selected lists",
) -> list[_ComparisonPanel]:
    """Arrange selected datasets for condition or dataset comparisons.

    Parameters
    ----------
    datasets : sequence of RecallDataset
        Nonempty ordered datasets. Their arrays are shared with the returned
        panels, without copying or modification.
    masks_by_dataset : sequence of sequences of Boolean arrays or None
        One ordered condition-mask sequence per dataset. Each mask selects trials
        within that dataset; None selects all trials.
    dataset_labels : sequence of str
        One label per dataset, in the same order.
    condition_labels : sequence of str
        Labels for the common ordered conditions.
    mode : Literal["conditions", "datasets"]
        "conditions" requires one dataset and compares its conditions in one
        panel. "datasets" creates a panel for each condition, comparing all
        datasets within it, even when there is only one dataset.
    selection_label : str, optional
        Title of the single panel in conditions mode.

    Returns
    -------
    list of dict
        Input arrangements containing datasets, trial_masks, labels and title.
        Callers may add analysis-specific options such as output position or
        category. Queries and figures are handled by the caller.

    Notes
    -----
    Each returned panel contains these fields:

    | Field | Contents |
    | --- | --- |
    | datasets | Ordered datasets to compare in this panel. |
    | trial_masks | One Boolean trial selection for each dataset entry. |
    | labels | One curve label for each dataset entry. |
    | title | The condition label, or selection_label in conditions mode. |

    In conditions mode, the same dataset appears once per condition with a
    different mask. In datasets mode, each panel contains all datasets with the
    corresponding condition masks. Mismatched numbers of datasets, labels,
    or condition masks raise ``ValueError``.
    """
    overlay_conditions = mode == "conditions"
    if overlay_conditions and len(datasets) != 1:
        raise ValueError("Choose one response dataset to overlay conditions.")

    titles = [selection_label] if overlay_conditions else condition_labels
    panels: list[_ComparisonPanel] = [
        {"title": title, "datasets": [], "labels": [], "trial_masks": []}
        for title in titles
    ]
    for dataset, dataset_label, masks in zip(
        datasets, dataset_labels, masks_by_dataset, strict=True
    ):
        selections = zip(condition_labels, masks, strict=True)
        for index, (condition_label, mask) in enumerate(selections):
            if overlay_conditions:
                panel, label = panels[0], condition_label
            else:
                panel, label = panels[index], dataset_label
            panel["datasets"].append(dataset)
            panel["labels"].append(label)
            panel["trial_masks"].append(
                jnp.ones(dataset["recalls"].shape[0], dtype=bool)
                if mask is None
                else jnp.asarray(mask)
            )
    return panels


def iter_analysis_figures(
    datasets: Sequence[RecallDataset],
    analysis_configs: Sequence[Mapping[str, Any]],
    labels: Sequence[str] | None = None,
    *,
    base_masks: Sequence[Bool[Array, " trials"] | None] | None = None,
    trial_query: str | None = None,
    confidence_level: float = 0.95,
    figure_size: tuple[float, float] = (10, 4.8),
    suffix: str = "",
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Iterator[tuple[Figure, str]]:
    """Create configured behavioral figures in order, one figure at a time.

    Parameters
    ----------
    datasets : sequence of RecallDataset
        Already-loaded observations or simulations.
    analysis_configs : sequence of mappings
        Each requires target: an analysis plotter callable or dotted import name.
        Optional entries are trial_query, figsize, labels, contrast_name, kwargs,
        color_cycle, title, ylim and figure_suffix. Plotters accept dataset/mask
        sequences and the usual analysis plot options, including axis.
    labels : sequence of str, optional
        Default labels, one per dataset. Configurations may override them.
    base_masks : sequence of Boolean arrays or None, optional
        Restrictions for each dataset. None selects all trials. Each analysis's
        query is intersected with these masks.
    trial_query : str, optional
        Default query, overridden by a configuration's trial_query.
    confidence_level : float, optional
        Default interval coverage; configuration kwargs may override it.
    figure_size : tuple of float, optional
        Default figure width and height in inches; figsize overrides it.
    suffix : str, optional
        Appended to the configured or target-derived figure suffix.

    Yields
    ------
    tuple of Figure and str
        Figure and filename suffix in configuration order. Only the next
        requested figure is created. The caller saves, displays and closes it.
        Repeated targets without figure_suffix receive an additional suffix
        containing their one-based configuration position.

    Notes
    -----
    kwargs override plot options except axis: this iterator always supplies axes
    belonging to the figure it yields. An explicit color_cycle overrides kwargs.
    Titles and limits are applied to the created axes after the plotter returns.
    Inputs are not modified. Import targets and queries are trusted executable
    code, not a restricted language.

    Configuration entries describe one figure:

    | Key | Meaning and default |
    | --- | --- |
    | target | Required plotter callable or dotted import name. |
    | trial_query | Query intersected with each base mask; defaults to the argument trial_query. An explicit None selects all trials within each base mask. |
    | figsize | Figure width and height in inches; defaults to figure_size. |
    | labels | Dataset labels; defaults to the labels argument, or empty labels when that argument is omitted. |
    | contrast_name | Legend title; defaults to Responses. |
    | kwargs | Additional plotter options, overriding labels, contrast_name and confidence_level when supplied. |
    | color_cycle | A non-None value overrides the corresponding entry in kwargs. |
    | title | Axes title; defaults to the plotter's name. |
    | ylim | Vertical limits, applied after plotting when non-None. |
    | figure_suffix | Filename stem; defaults to the plotter name without a leading plot_. The suffix argument is appended. |

    The axis option always belongs to the new figure, including when kwargs
    contains axis. Repeated targets with no explicit figure_suffix append their
    one-based configuration position after suffix. Explicit figure_suffix values
    are used as supplied; the caller is responsible for distinct output names.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    datasets = list(datasets)
    configurations = list(analysis_configs)
    labels = [""] * len(datasets) if labels is None else list(labels)
    if base_masks is None:
        base_masks = [None] * len(datasets)
    base_trial_masks = [
        jnp.ones(data["recalls"].shape[0], dtype=bool)
        if mask is None
        else jnp.asarray(mask)
        for data, mask in zip(datasets, base_masks)
    ]
    for index, config in enumerate(configurations):
        target = config["target"]
        plotter = import_from_string(target) if isinstance(target, str) else target
        plotter_name = getattr(plotter, "__name__", type(plotter).__name__)
        masks = [
            generate_trial_mask(data, config.get("trial_query", trial_query)) & base
            for data, base in zip(datasets, base_trial_masks)
        ]
        figure, axis = plt.subplots(figsize=config.get("figsize", figure_size), tight_layout=True)
        options = {
            "labels": config.get("labels", labels),
            "contrast_name": config.get("contrast_name", "Responses"),
            "confidence_level": confidence_level,
            "n_resamples": n_resamples,
            "bootstrap_seed": bootstrap_seed,
            **config.get("kwargs", {}),
            "axis": axis,
        }
        if config.get("color_cycle") is not None:
            options["color_cycle"] = config["color_cycle"]
        plotter(datasets, masks, **options)
        axis.set_title(config.get("title", plotter_name))
        if config.get("ylim") is not None:
            axis.set_ylim(config["ylim"])
        figure_suffix = (
            config.get("figure_suffix", plotter_name.removeprefix("plot_")) + suffix
        )
        if (
            sum(other["target"] == target for other in configurations) > 1
            and "figure_suffix" not in config
        ):
            figure_suffix += f"_{index + 1}"
        yield figure, figure_suffix


def save_figure(
    figure_dir: str | Path,
    figure_str: str,
    suffix: str | None = None,
    dpi: int = 600,
    formats: Sequence[str] = ("png",),
    *,
    figure: Figure,
) -> None:
    """Lay out and save a supplied figure without displaying or closing it.

    Parameters
    ----------
    figure_dir : str or pathlib.Path
        Output directory, created with its parents when necessary.
    figure_str : str
        Nonempty base filename without an extension. Callers that only want to
        display a figure should skip saving and call pyplot.show themselves.
    suffix : str or None, optional
        Nonempty suffixes are appended to the base filename with an underscore.
    dpi : int, optional
        Saved image resolution. Defaults to 600.
    formats : sequence of str, optional
        Matplotlib formats such as png, svg or pdf. Defaults to png; repeated
        formats are written only once, preserving their first-occurrence order.
    figure : matplotlib.figure.Figure
        Figure to save. It receives tight_layout before writing, and each
        format uses bbox_inches="tight". Existing destination files are replaced.

    Returns
    -------
    None
        The figure remains open. No current figure is looked up or displayed.
    """
    figure.tight_layout()
    output_directory = Path(figure_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename_suffix = f"_{suffix}" if suffix else ""
    for image_format in dict.fromkeys(formats):
        figure_path = output_directory / f"{figure_str}{filename_suffix}.{image_format}"
        figure.savefig(figure_path, bbox_inches="tight", dpi=dpi)
