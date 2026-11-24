"""Proportion of eligible presentations recalled within a category.

Suppose fruit appeared first on two lists and third on eight. If one of the
first-position fruits and three of the third-position fruits were recalled,
``cat_recall_rate`` gives 4/10 for fruit. ``cat_spc`` keeps the two rates,
1/2 and 3/8, separate by study position.

Choose the category field and value to select presentations. An optional
validity field can restrict them further. Each eligible occurrence of a
recalled item earns credit once. A category with no eligible presentations
has an undefined rate. ``subject_cat_recall_rate`` keeps a row per participant
and a column per requested category.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jax import jit
from jax import numpy as jnp
from matplotlib import rcParams
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, prepare_plot_inputs
from ..typing import Array, Bool, Float, Integer, RecallDataset
from ._study_positions import recall_status, study_position_mask
from .statistics import mean_interval

__all__ = [
    "cat_recall_rate",
    "category_recall_hits",
    "plot_cat_recall_rate",
    "plot_category_rates",
    "subject_cat_recall_rate",
]


def category_recall_hits(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    valid: Bool[Array, " study_positions"],
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recalled-position indicators restricted to a category."""
    recalled = jnp.bincount(recalls, length=list_length + 1)[1:] > 0
    matches = (categories == category_value) & valid
    return recalled.astype(jnp.float32) * matches.astype(jnp.float32)


def cat_recall_rate(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    valid_field: str | None = None,
) -> Float[Array, ""]:
    """Category-filtered scalar recall rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item category metadata.
    category_field : str
        Key providing categories aligned with the ``pres_itemnos`` array.
    category_value : int
        Category to filter on.
    valid_field : str, optional
        Key with positive values at valid study positions. If ``None``,
        all actual presentations are eligible. Padding is always excluded.

    Returns
    -------
    Float
        Proportion of eligible target-category presentations whose item was recalled.
        Every occurrence of a recalled item contributes once; responses are not
        counted repeatedly. An absent category returns NaN.

    """
    eligible = study_position_mask(dataset, valid_field) & (
        dataset[category_field] == category_value
    )
    return (recall_status(dataset) & eligible).sum() / eligible.sum()


def subject_cat_recall_rate(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    valid_field: str | None = None,
) -> Float[np.ndarray, "subjects categories"]:
    """Return subject-level category recall rates.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item category metadata.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    category_field : str
        Key providing categories aligned with the ``pres_itemnos`` array.
    category_values : Sequence[int]
        Category values to summarize.
    valid_field : str, optional
        Key with positive values at valid study positions.

    Returns
    -------
    np.ndarray
        Subject by category matrix of recall proportions.

    """
    columns = []
    for category_value in category_values:
        subject_values = apply_by_subject(
            dataset,
            trial_mask,
            jit(
                cat_recall_rate,
                static_argnames=("category_field", "category_value", "valid_field"),
            ),
            category_field=category_field,
            category_value=category_value,
            valid_field=valid_field,
        )
        columns.append(np.asarray(subject_values, dtype=float))

    return np.stack(columns, axis=1) if columns else np.empty((0, 0), dtype=float)


def _category_point_summaries(
    subject_values: Float[np.ndarray, " subjects categories"],
    confidence_level: float,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> list[tuple[int, float, list[list[float]]]]:
    """Summarize available categories in their original bootstrap order.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    summaries = []
    for category_index in range(subject_values.shape[1]):
        valid_values = subject_values[:, category_index]
        valid_values = valid_values[np.isfinite(valid_values)]
        if not valid_values.size:
            continue
        interval = mean_interval(
            valid_values,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
        )
        mean = interval["mean"]
        errors = [[mean - interval["low"]], [interval["high"] - mean]]
        summaries.append((category_index, mean, errors))
    return summaries


def plot_category_rates(
    axis: Axes,
    rates: Sequence[Float[np.ndarray, "participants categories"]],
    labels: Sequence[str],
    category_labels: Sequence[str],
    *,
    color_cycle: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
    contrast_name: str | None = None,
) -> Axes:
    """Plot category means from participant rates already calculated for each group.

    Parameters
    ----------
    axis : Axes
        Axes on which to draw the groups.
    rates : sequence of arrays
        One participant-by-category matrix per group, as returned by
        ``subject_cat_recall_rate``. Each finite participant rate has equal
        weight. A category missing from a group leaves no point in that group.
    labels, category_labels : sequences of str
        Horizontal-axis group names and category names in column order.
    color_cycle : sequence of str, optional
        Category colors, shared across groups.
    confidence_level, n_resamples, bootstrap_seed : optional
        Coverage, number of bootstrap draws, and random seed for the interval
        around each category mean.
    contrast_name : str, optional
        Title for the category legend.

    Returns
    -------
    Axes
        The axes containing the category means and their intervals.
    """
    if color_cycle is None:
        color_cycle = [entry["color"] for entry in rcParams["axes.prop_cycle"]]
    x = np.arange(len(rates))
    dodge_width = 0.35
    labeled_category_indices = set()

    for data_index, subject_values_for_dataset in enumerate(rates):
        summaries = _category_point_summaries(
            subject_values_for_dataset,
            confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        if not summaries:
            continue
        offsets = (
            np.array([0.0])
            if len(summaries) == 1
            else np.linspace(-dodge_width / 2, dodge_width / 2, len(summaries))
        )
        for present_index, (category_index, mean, yerr) in enumerate(summaries):
            offset = offsets[present_index]

            color = color_cycle[category_index % len(color_cycle)]
            label = None
            if category_index not in labeled_category_indices:
                label = category_labels[category_index]
                labeled_category_indices.add(category_index)

            axis.errorbar(
                x[data_index] + offset,
                mean,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor="black",
                capsize=5,
                linestyle="none",
                markersize=6,
                label=label,
                zorder=3,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    format_analysis_axes(axis, "", "Recall Rate", contrast_name)
    return axis


def plot_cat_recall_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    valid_field: str | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    category_labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot category-filtered recall rates as points with error bars.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    category_field : str
        Key providing categories aligned with the ``pres_itemnos`` array.
    category_values : Sequence[int]
        Categories to plot.
    valid_field : str, optional
        Key with positive values at valid study positions.
    color_cycle : list[str], optional
        Colors for categories.
    labels : Sequence[str], optional
        X-axis labels for datasets or trial masks.
    category_labels : Sequence[str], optional
        Legend labels for category values.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Axes with the category recall-rate plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)
    if category_labels is None:
        category_labels = [str(value) for value in category_values]

    subject_values_by_dataset = [
        subject_cat_recall_rate(
            data,
            trial_masks[data_index],
            category_field,
            category_values,
            valid_field=valid_field,
        )
        for data_index, data in enumerate(datasets)
    ]
    return plot_category_rates(
        axis, subject_values_by_dataset, labels, category_labels,
        color_cycle=color_cycle, confidence_level=confidence_level,
        n_resamples=n_resamples, bootstrap_seed=bootstrap_seed,
        contrast_name=contrast_name,
    )
