"""Joint and conditional recall of pairs of different study items.

``corec_by_lag`` measures the probability that both items in an eligible pair
were recalled. ``conditional_corec_by_lag`` conditions on recalling the earlier
study position's item. ``conditional_corec_by_cat`` instead selects recalled
anchors by category and compares same-category with different-category targets.

Responses identify items through study positions. A recalled item credits all
of its presentations, but two occurrences of that same item never form an
eligible pair. Padding is excluded. Dataset calculations pool pair counts;
plotters calculate each participant's probability before averaging participants.
Zero-opportunity groups remain undefined.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, RecallDataset
from ._study_positions import recall_status, study_position_mask

__all__ = [
    "conditional_corec_by_cat",
    "conditional_corec_by_lag",
    "corec_by_lag",
    "plot_conditional_corec_by_cat",
    "plot_conditional_corec_by_lag",
    "plot_corec_by_lag",
]


def _eligible_pairs(
    dataset: RecallDataset,
) -> Bool[Array, " trials study_positions study_positions"]:
    """Pairs of actual presentations belonging to different items."""
    included = study_position_mask(dataset)
    items = dataset["pres_itemnos"]
    return (
        included[:, :, None]
        & included[:, None, :]
        & (items[:, :, None] != items[:, None, :])
    )


def _counts_by_lag(
    pairs: Bool[Array, " trials study_positions study_positions"],
) -> Float[Array, " lags"]:
    """Sum ordered pair indicators at each positive study-position lag."""
    positions = jnp.arange(pairs.shape[1])
    offsets = positions[None, :] - positions[:, None]
    totals = pairs.sum(axis=0)
    return vmap(lambda lag: jnp.sum(jnp.where(offsets == lag, totals, 0)))(
        positions[1:]
    )


def corec_by_lag(dataset: RecallDataset) -> Float[Array, " lags"]:
    """Return CoRec(d) for positive lags.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and
        ``pres_itemnos``.

    Returns
    -------
    Float[Array, " lags"]
        Co-recall probability at each positive lag.

    """

    possible = _eligible_pairs(dataset)
    recalled = recall_status(dataset)
    actual = possible & recalled[:, :, None] & recalled[:, None, :]
    return _counts_by_lag(actual) / _counts_by_lag(possible)


def plot_corec_by_lag(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot co-recall curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset or list of datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials within each
        dataset.
    max_lag : int, optional
        Maximum lag to display (defaults to full
        range).
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing Matplotlib ``Axes`` to plot on.
    confidence_level : float, optional
        Confidence level for the bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with co-recall curves.
    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    for index, data in enumerate(datasets):
        lag_count = data["pres_itemnos"].shape[1] - 1
        lag_limit = lag_count if max_lag is None else min(max_lag, lag_count)
        lag_axis = jnp.arange(1, lag_limit + 1, dtype=int)
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                jit(corec_by_lag),
            )
        )
        subject_values = subject_values[:, :lag_limit]

        color = color_cycle[index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_axis,
            subject_values,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(axis, "Lag", "Co-Recall Probability", contrast_name)
    return axis


def conditional_corec_by_lag(dataset: RecallDataset) -> Float[Array, " lags"]:
    """Return conditional CoRec(d) for positive lags.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and
        ``pres_itemnos``.

    Returns
    -------
    Float[Array, " lags"]
        Conditional co-recall probability at each
        positive lag.

    """

    recalled = recall_status(dataset)
    possible = _eligible_pairs(dataset) & recalled[:, :, None]
    actual = possible & recalled[:, None, :]
    return _counts_by_lag(actual) / _counts_by_lag(possible)


def plot_conditional_corec_by_lag(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot conditional co-recall curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset or list of datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials within each
        dataset.
    max_lag : int, optional
        Maximum lag to display (defaults to full range).
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing Matplotlib ``Axes`` to plot on.
    confidence_level : float, optional
        Confidence level for the bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with conditional co-recall
        curves.
    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    for index, data in enumerate(datasets):
        lag_count = data["pres_itemnos"].shape[1] - 1
        lag_limit = lag_count if max_lag is None else min(max_lag, lag_count)
        lag_axis = jnp.arange(1, lag_limit + 1, dtype=int)
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                jit(conditional_corec_by_lag),
            )
        )
        subject_values = subject_values[:, :lag_limit]

        color = color_cycle[index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_axis,
            subject_values,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(
        axis, "Lag", "Conditional Co-Recall Probability", contrast_name
    )
    return axis


def conditional_corec_by_cat(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
) -> Float[Array, " relation"]:
    """Return conditional category-wise co-recall probabilities.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and
        the requested category field.
    category_field : str
        Key mapping to per-trial study-item
        categories.
    category_value : int
        Reference category defining the anchor items.

    Returns
    -------
    Float[Array, " relation"]
        Same- and different-category conditional
        co-recall probabilities.

    """

    recalled = recall_status(dataset)
    category = dataset[category_field] == category_value
    possible = _eligible_pairs(dataset) & (recalled & category)[:, :, None]
    actual = possible & recalled[:, None, :]
    same = category[:, None, :]
    numerator = jnp.stack([(actual & same).sum(), (actual & ~same).sum()])
    denominator = jnp.stack([(possible & same).sum(), (possible & ~same).sum()])
    return numerator / denominator


def plot_conditional_corec_by_cat(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_value: int,
    relation_labels: Sequence[str] = ("Same", "Different"),
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot conditional co-recall by category with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset or list of datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials within each
        dataset.
    category_field : str
        Dataset key containing per-study-item
        categories.
    category_value : int
        Reference category defining the anchors.
    relation_labels : Sequence[str], optional
        Tick labels for same- and different-category
        neighbor probabilities.
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing Matplotlib ``Axes`` to plot on.
    confidence_level : float, optional
        Confidence level for the bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with conditional co-recall
        by category curves.
    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    relation_axis = jnp.arange(len(relation_labels), dtype=int)

    for index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                jit(
                    conditional_corec_by_cat,
                    static_argnames=("category_field", "category_value"),
                ),
                category_field=category_field,
                category_value=category_value,
            )
        )

        color = color_cycle[index % len(color_cycle)]
        plot_mean_curve(
            axis,
            relation_axis,
            subject_values,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    axis.set_xticks(relation_axis)
    axis.set_xticklabels(list(relation_labels))

    format_analysis_axes(
        axis,
        "Category Relation",
        "Conditional Co-Recall Probability",
        contrast_name,
    )
    return axis
