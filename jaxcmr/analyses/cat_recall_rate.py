"""Category-filtered recall-rate analyses.

Computes scalar recall proportions separately for each item category.
This is the category-level summary analogue of category SPC analyses:
it is parameterized by a category field and category values rather than
being tied to any particular experimental contrast.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy import stats
from scipy.stats import bootstrap

from ..helpers import apply_by_subject
from ..plotting import prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = [
    "category_recall_hits",
    "cat_recall_rate",
    "subject_cat_recall_rate",
    "plot_cat_recall_rate",
    "CatRecallRatePairTestResult",
    "test_cat_recall_rate_pair",
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
    valid_field: Optional[str] = None,
) -> Float:
    """Category-filtered scalar recall rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item category metadata.
    category_field : str
        Key providing item categories per study position.
    category_value : int
        Category to filter on.
    valid_field : str, optional
        Key with positive values at valid study positions. If ``None``,
        all category positions are treated as valid.

    Returns
    -------
    Float
        Proportion of valid target-category positions that were recalled.

    """
    recalls = dataset["recalls"]
    categories = dataset[category_field]
    list_length = categories.shape[1]
    if valid_field is None:
        valid = jnp.ones_like(categories, dtype=bool)
    else:
        valid = dataset[valid_field] > 0

    recall_hits = vmap(
        category_recall_hits,
        in_axes=(0, 0, None, 0, None),
    )(recalls, categories, category_value, valid, list_length)

    denominator = jnp.sum((categories == category_value) & valid)
    numerator = jnp.sum(recall_hits)
    return jnp.where(
        denominator > 0,
        numerator / denominator,
        jnp.float32(jnp.nan),
    )


def subject_cat_recall_rate(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    valid_field: Optional[str] = None,
) -> np.ndarray:
    """Return subject-level category recall rates.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with per-item category metadata.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    category_field : str
        Key providing item categories per study position.
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

    if not columns:
        return np.empty((0, 0), dtype=float)
    return np.stack(columns, axis=1)


def plot_cat_recall_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    valid_field: Optional[str] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    category_labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot category-filtered recall rates as points with error bars.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    category_field : str
        Key providing item categories per study position.
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

    x = np.arange(len(datasets))
    dodge_width = 0.35
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
    labeled_category_indices = set()

    for data_index, subject_values_for_dataset in enumerate(subject_values_by_dataset):
        present_category_indices = [
            category_index
            for category_index in range(len(category_values))
            if np.any(np.isfinite(subject_values_for_dataset[:, category_index]))
        ]
        if not present_category_indices:
            continue

        offsets = (
            np.array([0.0])
            if len(present_category_indices) == 1
            else np.linspace(
                -dodge_width / 2,
                dodge_width / 2,
                len(present_category_indices),
            )
        )
        for present_index, category_index in enumerate(present_category_indices):
            offset = offsets[present_index]
            valid_values = subject_values_for_dataset[:, category_index]
            valid_values = valid_values[np.isfinite(valid_values)]
            mean = float(np.nanmean(valid_values))
            if len(valid_values) > 1:
                ci = bootstrap(
                    (valid_values,),
                    np.nanmean,
                    confidence_level=confidence_level,
                ).confidence_interval
                yerr = [[mean - ci.low], [ci.high - mean]]
            else:
                yerr = [[0], [0]]

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
    set_plot_labels(axis, "", "Recall Rate", contrast_name)
    return axis


@dataclass
class CatRecallRatePairTestResult:
    """Results from a paired category recall-rate comparison."""

    left_label: str
    right_label: str
    n: int
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float
    mean_diff: float

    def __str__(self) -> str:
        lines = [
            f"Comparison: {self.left_label} - {self.right_label}",
            f"n = {self.n}",
            f"Mean Diff = {self.mean_diff:.4f}",
            f"t-stat = {self.t_stat:.3f}, t p-val = {self.t_pval:.4f}",
            f"W-stat = {self.w_stat:.1f}, W p-val = {self.w_pval:.4f}",
        ]
        return "\n".join(lines)


def test_cat_recall_rate_pair(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "left",
    right_label: str = "right",
) -> CatRecallRatePairTestResult:
    """Test paired subject-level category recall rates."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left_valid = left[valid]
    right_valid = right[valid]
    diff = left_valid - right_valid

    if len(diff) > 1:
        t_stat, t_pval = stats.ttest_rel(left_valid, right_valid, nan_policy="omit")
    else:
        t_stat, t_pval = np.nan, np.nan

    if len(diff) > 0:
        try:
            w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = np.nan, np.nan
    else:
        w_stat, w_pval = np.nan, np.nan

    return CatRecallRatePairTestResult(
        left_label=left_label,
        right_label=right_label,
        n=int(len(diff)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat),
        w_pval=float(w_pval),
        mean_diff=float(np.nanmean(diff)) if len(diff) > 0 else float("nan"),
    )
