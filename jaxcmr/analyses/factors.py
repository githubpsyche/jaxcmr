"""Measurements associated with study items, and recall probability by measurement.

Use ``cat_factor_spc`` for measurement means at each study position,
``cat_factor_by_recall`` to separate those means by whether the item was
recalled, and ``cat_recall_by_factor`` for recall probability within measurement
bins. Fields and category values are explicit inputs; the operations do not
assume a particular experiment or measurement unit.

Dataset calculations pool eligible presentations. Their ``subject_*``
counterparts keep participants separate, and plotters average participant
summaries with equal weight. Missing measurements and optional quality flags
exclude observations, while recall status credits every occurrence of a
recalled item. Shared bins keep dataset comparisons on the same scale.
"""

from collections.abc import Callable, Sequence

import numpy as np
from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, Integer, Real, RecallDataset
from ._study_positions import recall_status, recalled_positions, study_position_mask

__all__ = [
    "binned_recall_counts",
    "cat_factor_by_recall",
    "cat_factor_spc",
    "cat_recall_by_factor",
    "category_factor_recall_histogram",
    "category_factor_values",
    "compute_factor_bin_edges",
    "factor_sums_and_counts",
    "plot_cat_factor_by_recall",
    "plot_cat_factor_spc",
    "plot_cat_recall_by_factor",
    "subject_cat_factor_by_recall",
    "subject_cat_factor_spc",
    "subject_cat_recall_by_factor",
]


def factor_sums_and_counts(
    values: Real[Array, " trials study_positions"],
    include: Bool[Array, " trials study_positions"],
) -> tuple[Float[Array, " study_positions"], Integer[Array, " study_positions"]]:
    """Sum finite included measurements and count them at each study position.

    Parameters
    ----------
    values : array of shape (trials, study_positions)
        Numeric measurements. Integer values are promoted to floating point.
    include : boolean array of shape (trials, study_positions)
        Observations to include; nonfinite measurements are additionally excluded.

    Returns
    -------
    tuple of arrays of shape (study_positions,)
        Measurement sums and observation counts. Empty positions have zero counts
        and sums; callers report their means as undefined. Supports JIT.
    """
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    include = jnp.asarray(include, dtype=bool)
    valid = include & jnp.isfinite(values)
    return jnp.sum(jnp.where(valid, values, 0), axis=0), jnp.sum(valid, axis=0)


def category_factor_values(
    factor: Real[Array, " trials study_positions"],
    categories: Integer[Array, " trials study_positions"],
    category_value: int,
    *,
    valid: Bool[Array, " trials study_positions"] | None = None,
) -> Float[Array, " study_positions"]:
    """Average finite measurements for one category at each study position.

    Parameters
    ----------
    factor : array of shape (trials, study_positions)
        Measurements to average across trials.
    categories : integer array of shape (trials, study_positions)
        Category at each presentation.
    category_value : int
        Category to include, including zero if it is a meaningful category code.
    valid : boolean array of shape (trials, study_positions), optional
        Additional inclusion mask. Omit to use all finite values in the category.

    Returns
    -------
    array of shape (study_positions,)
        Means, with NaN for positions without eligible observations. Each included
        trial contributes equally. Supports JIT.
    """
    include = categories == category_value
    if valid is not None:
        include = include & valid
    sums, counts = factor_sums_and_counts(factor, include)
    return jnp.where(counts > 0, sums / jnp.maximum(counts, 1), jnp.nan)


def cat_factor_spc(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    factor_field: str,
    *,
    valid_field: str | None = None,
) -> Float[Array, " study_positions"]:
    """Return mean measurements by study position within one category.

    A "factor" is a numeric measurement attached to each presentation, such as
    LPP amplitude. ``factor_field`` and ``category_field`` name arrays shaped
    (trials, study_positions), aligned with ``pres_itemnos``. Recalls are not used.

    At each position, average finite measurements from the selected category
    across trials. Exclude padding, presentations after the declared event count, and any
    nonfinite or nonpositive flags in the optional ``valid_field``. Return one
    value per study position; positions with no eligible measurements are NaN.

    This pools observations. Use ``subject_cat_factor_spc`` for separate
    participant means and its plotting wrapper for equal-participant weighting.
    JIT requires field names static; the category value may vary dynamically.
    """
    measurements = dataset[factor_field]
    categories = dataset[category_field]
    included = study_position_mask(dataset, valid_field)
    return category_factor_values(
        measurements, categories, category_value, valid=included
    )


def subject_cat_factor_spc(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    *,
    valid_field: str | None = None,
) -> Float[Array, " subjects categories study_positions"]:
    """Return separate subject means, sorted by ID and in requested category order.

    Subjects without selected trials are omitted. Empty selections return a
    zero-length subject axis. Averaging this axis with ``jnp.nanmean`` gives
    equal weight to subjects with data in each cell. Grouping runs outside JIT;
    the within-subject calculation is compiled.
    """
    categories = tuple(category_values)

    @jit
    def summarize(data: RecallDataset) -> Float[Array, " categories study_positions"]:
        return jnp.stack(
            [
                cat_factor_spc(
                    data,
                    category_field,
                    category,
                    factor_field,
                    valid_field=valid_field,
                )
                for category in categories
            ]
        )

    results = apply_by_subject(dataset, trial_mask, summarize)
    if not results:
        return jnp.empty((0, len(categories), dataset["pres_itemnos"].shape[1]))
    return jnp.stack(results)


def _plot_factor_curves(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trials"]] | Bool[Array, " trials"],
    summarize: Callable[
        [RecallDataset, Bool[Array, " trials"]],
        Float[Array, " subjects curves study_positions"],
    ],
    curve_labels: Sequence[str],
    factor_label: str,
    *,
    color_cycle: list[str] | None,
    labels: Sequence[str] | None,
    contrast_name: str | None,
    axis: Axes | None,
    confidence_level: float,
    draw_ci: bool,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Draw subject-by-curve-by-position summaries with shared display conventions.

    Supply nonempty datasets, curves and colors, one mask and dataset label per dataset,
    and one curve label per curve.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [
            f"Dataset {i + 1}" if len(datasets) > 1 else ""
            for i in range(len(datasets))
        ]
    for data_index, (data, mask) in enumerate(zip(datasets, trial_masks)):
        values = summarize(data, mask)
        if values.shape[0] == 0:
            continue
        positions = jnp.arange(1, values.shape[-1] + 1)
        for curve_index, curve_label in enumerate(curve_labels):
            label = " / ".join(
                part for part in (labels[data_index], curve_label) if part
            )
            color_index = data_index * len(curve_labels) + curve_index
            plot_mean_curve(
                axis,
                positions,
                values[:, curve_index, :],
                label,
                color_cycle[color_index % len(color_cycle)],
                draw_ci=draw_ci,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
    format_analysis_axes(axis, "Study Position", factor_label, contrast_name)
    return axis


def plot_cat_factor_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trials"]] | Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
    category_labels: Sequence[str] | None = None,
    factor_label: str | None = None,
    draw_ci: bool = True,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot equal-participant-weighted measurement means by study position.

    Each curve first averages the finite eligible measurements within each
    participant, following ``cat_factor_spc``, then weights available participant
    means equally. This displays measurement values, not recall probabilities.

    ``labels`` names datasets; ``category_labels`` names categories. Axis text
    defaults to ``factor_field``; ``factor_label`` can add a description/units.
    ``draw_ci`` controls interval display without changing the subject means.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    categories = tuple(category_values)
    if category_labels is None:
        category_labels = [
            str(value) if len(categories) > 1 else "" for value in categories
        ]

    def summarize(
        data: RecallDataset,
        mask: Bool[Array, " trials"],
    ) -> Float[Array, " subjects curves study_positions"]:
        return subject_cat_factor_spc(
            data,
            mask,
            category_field,
            categories,
            factor_field,
            valid_field=valid_field,
        )

    return _plot_factor_curves(
        datasets,
        trial_masks,
        summarize,
        category_labels,
        factor_field if factor_label is None else factor_label,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        draw_ci=draw_ci,
    )


def cat_factor_by_recall(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    factor_field: str,
    *,
    valid_field: str | None = None,
) -> Float[Array, "2 study_positions"]:
    """Return measurement means with axes (recall_status, study_position).

    ``factor_field`` selects a numeric measurement per presentation, such as
    LPP amplitude; it and ``category_field`` align with ``pres_itemnos``. Rows
    are [recalled, not recalled], with means pooled across eligible trials.

    All eligible presentations of a recalled item enter the recalled group.
    A response position identifies the item, not which occurrence was remembered.
    Duplicate responses count once; invalid codes and padding add no hits.

    Within each category/status/position group, average finite measurements at
    actual presentations. Optional ``valid_field`` flags must be finite and
    positive. Missing measurements and quality failures exclude observations
    from means without changing whether the item was recalled.
    Empty groups are NaN. Use ``subject_cat_factor_by_recall`` for separate
    participant means rather than pooling trials.

    Field names must be static under JIT.
    """
    measurements = dataset[factor_field]
    categories = dataset[category_field]
    included = study_position_mask(dataset, valid_field)
    recalled = recall_status(dataset)
    return jnp.stack(
        [
            category_factor_values(
                measurements, categories, category_value, valid=included & recalled
            ),
            category_factor_values(
                measurements, categories, category_value, valid=included & ~recalled
            ),
        ]
    )


def subject_cat_factor_by_recall(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    *,
    valid_field: str | None = None,
) -> Float[Array, " subjects categories 2 study_positions"]:
    """Return subject-by-category-by-status-by-position factor means.

    Subject IDs are sorted; category order follows the caller; statuses are
    [recalled, not recalled]. Subjects without selected trials are omitted.
    Empty selections return a zero-length subject axis. Grouping runs outside
    JIT; each subject calculation is compiled. Averaging the subject axis with
    ``jnp.nanmean`` weights available subjects equally in each cell.
    """
    categories = tuple(category_values)

    @jit
    def summarize(data: RecallDataset) -> Float[Array, " categories 2 study_positions"]:
        return jnp.stack(
            [
                cat_factor_by_recall(
                    data,
                    category_field,
                    category,
                    factor_field,
                    valid_field=valid_field,
                )
                for category in categories
            ]
        )

    results = apply_by_subject(dataset, trial_mask, summarize)
    if not results:
        return jnp.empty((0, len(categories), 2, dataset["pres_itemnos"].shape[1]))
    return jnp.stack(results)


def plot_cat_factor_by_recall(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trials"]] | Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
    category_labels: Sequence[str] | None = None,
    recall_labels: Sequence[str] = ("Recalled", "Not recalled"),
    factor_label: str | None = None,
    draw_ci: bool = True,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot equal-participant-weighted measurement means split by recall status.

    Recall classification and observation inclusion follow ``cat_factor_by_recall``.
    The default labels "Recalled" and "Not recalled" refer to items: every
    eligible presentation of a recalled item enters its recalled group.
    ``recall_labels`` changes the displayed labels, not this classification.

    ``labels`` names datasets. Category and recall labels identify each curve.
    Participant means receive equal weight wherever available. Hiding confidence
    intervals never changes the averaging or input dataset. Measurement units
    can be supplied through ``factor_label``.

    Supply nonempty categories, one category label per category, and two recall-status
    labels (recalled, not recalled).

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    categories = tuple(category_values)
    if category_labels is None:
        category_labels = [
            str(value) if len(categories) > 1 else "" for value in categories
        ]
    curve_labels = [
        " / ".join(part for part in (category_label, recalled_label) if part)
        for category_label in category_labels
        for recalled_label in recall_labels
    ]

    def summarize(
        data: RecallDataset,
        mask: Bool[Array, " trials"],
    ) -> Float[Array, " subjects curves study_positions"]:
        values = subject_cat_factor_by_recall(
            data,
            mask,
            category_field,
            categories,
            factor_field,
            valid_field=valid_field,
        )
        return values.reshape(values.shape[0], len(categories) * 2, values.shape[-1])

    return _plot_factor_curves(
        datasets,
        trial_masks,
        summarize,
        curve_labels,
        factor_field if factor_label is None else factor_label,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        draw_ci=draw_ci,
    )


def compute_factor_bin_edges(
    values: Real[Array, " ..."],
    valid: Bool[Array, " ..."],
    bin_count: int = 10,
) -> Float[Array, " bin_count_plus_one"]:
    """Return equal-width edges from finite, included reference values.

    ``valid`` must match ``values``. No eligible values gives all-NaN edges.
    Constant values get a symmetric interval with half-width
    ``max(0.5, abs(value) * 0.01)`` so the edges remain distinct at large scales.
    This helper supports JIT with a positive, static integer ``bin_count``.
    It does not inspect recall outcomes or infer which datasets/categories
    should define the bins.
    """
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    valid = jnp.asarray(valid, dtype=bool)
    included = valid & jnp.isfinite(values)
    lower = jnp.min(jnp.where(included, values, jnp.inf), initial=jnp.inf)
    upper = jnp.max(jnp.where(included, values, -jnp.inf), initial=-jnp.inf)
    half_width = jnp.maximum(0.5, jnp.abs(lower) * 0.01)
    constant = lower == upper
    edges = jnp.linspace(
        jnp.where(constant, lower - half_width, lower),
        jnp.where(constant, upper + half_width, upper),
        bin_count + 1,
    )
    return jnp.where(jnp.any(included), edges, jnp.nan)


def binned_recall_counts(
    recalled: Bool[Array, " ..."],
    values: Real[Array, " ..."],
    include: Bool[Array, " ..."],
    bin_edges: Real[Array, " bin_count_plus_one"],
) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
    """Return recalled and eligible counts for fixed bins, in that order.

    Inputs share a shape, with one Boolean recall indicator per observation.
    Missing/nonfinite and out-of-range values contribute to neither count.
    Bins are left-closed/right-open, except the final bin includes its right
    edge. Empty bins have zero counts; rates derived from them are undefined.

    Supply a one-dimensional array of at least two finite, strictly
    increasing edges. Numerical calls assume aligned inputs and valid edges.
    All-NaN edges from an empty reference set yield zero counts.
    """
    values = jnp.asarray(values)
    recalled = jnp.asarray(recalled, dtype=bool)
    include = jnp.asarray(include, dtype=bool)
    edges = jnp.asarray(bin_edges)
    bin_count = edges.size - 1
    eligible = (
        include & jnp.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    )
    indices = jnp.searchsorted(edges, values, side="right") - 1
    indices = jnp.where(values == edges[-1], bin_count - 1, indices)

    # An extra bin absorbs excluded observations without dynamic indexing.
    def count(mask: Bool[Array, "trials study_positions"]) -> Integer[Array, " bins"]:
        selected = jnp.where(mask, indices, bin_count).reshape(-1)
        return jnp.bincount(selected, length=bin_count + 1)[:-1]

    return count(eligible & recalled), count(eligible)


def category_factor_recall_histogram(
    recalls: Integer[Array, " trials recall_events"],
    factor: Real[Array, " trials study_positions"],
    categories: Integer[Array, " trials study_positions"],
    category_value: int,
    bin_edges: Real[Array, " bin_count_plus_one"],
    *,
    valid: Bool[Array, " trials study_positions"] | None = None,
) -> Float[Array, " bins"]:
    """Pool recalled-position indicators within a category and fixed bins.

    This array interface identifies recalled positions directly; it has no item
    identities with which to expand repeated presentations. Use the dataset
    operation ``cat_recall_by_factor`` for repeated-item experiments.

    Repeated responses count once; zero, negative and out-of-range recall codes
    do not mark any position. ``valid`` excludes unusable study positions from
    both numerator and denominator. Empty bins return NaN, never zero.
    """
    if valid is None:
        valid = jnp.ones(factor.shape, dtype=bool)
    recalled = recalled_positions(recalls, factor.shape[1])
    numerator, denominator = binned_recall_counts(
        recalled, factor, valid & (categories == category_value), bin_edges
    )
    return jnp.where(denominator > 0, numerator / denominator, jnp.nan)


def cat_recall_by_factor(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    factor_field: str,
    bin_edges: Real[Array, " bin_count_plus_one"] | None = None,
    bin_count: int = 10,
    *,
    valid_field: str | None = None,
) -> Float[Array, " bins"]:
    """Return recall probabilities within bins of a presentation measurement.

    A "factor" is a numeric measurement per presentation, such as LPP amplitude.
    ``factor_field`` and ``category_field`` select trial-by-study-position arrays
    aligned with ``pres_itemnos``. The output has one recall probability per bin,
    not a mean measurement as in ``cat_factor_by_recall``.

    For the chosen category, divide recalled-position counts by eligible-position
    counts in each bin. Recall codes name study positions; duplicate responses
    count once. All eligible presentations of a recalled item contribute to recall counts.
    Exclude padding, presentations after the declared event count, nonfinite measurements,
    and values outside the bins from both counts. Optional ``valid_field`` flags
    must be finite and positive. Empty bins return NaN. Bins include their left
    edge; only the final bin also includes its right edge.

    Automatic edges use finite, eligible measurements across all categories in
    this dataset. For comparisons, compute reference edges once with
    ``compute_factor_bin_edges`` and pass them to each call. This function pools
    positions; ``subject_cat_recall_by_factor`` returns participant rates that
    can be averaged with equal participant weight in each available bin.

    JIT requires field names and ``bin_count`` static. The category and explicit
    bin-edge values may vary dynamically at a fixed shape.
    """
    measurements = dataset[factor_field]
    categories = dataset[category_field]
    included = study_position_mask(dataset, valid_field)
    if bin_edges is None:
        bin_edges = compute_factor_bin_edges(measurements, included, bin_count)
    recalled, possible = binned_recall_counts(
        recall_status(dataset),
        measurements,
        included & (categories == category_value),
        bin_edges,
    )
    return recalled / possible


def subject_cat_recall_by_factor(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    bin_edges: Real[Array, " bin_count_plus_one"],
    *,
    valid_field: str | None = None,
) -> Float[Array, " subjects categories bins"]:
    """Return subject rates using caller-supplied common edges.

    Subjects occur in ascending ID order, omitting those without selected
    trials. Category order follows ``category_values``. An empty selection
    returns shape ``(0, n_categories, n_bins)``. Each subject pools its own
    eligible positions; ``jnp.nanmean(result, axis=0)`` then weights subjects
    equally within each available bin. This grouping wrapper runs outside JIT;
    each subject's numerical calculation is compiled.
    """
    edges = jnp.asarray(bin_edges)
    categories = tuple(category_values)

    @jit
    def summarize(data: RecallDataset) -> Float[Array, "categories bins"]:
        return jnp.stack(
            [
                cat_recall_by_factor(
                    data,
                    category_field,
                    category,
                    factor_field,
                    edges,
                    valid_field=valid_field,
                )
                for category in categories
            ]
        )

    values = apply_by_subject(dataset, trial_mask, summarize)
    if not values:
        return jnp.empty((0, len(categories), edges.size - 1))
    return jnp.stack(values)


def plot_cat_recall_by_factor(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trials"]] | Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    bin_edges: Real[Array, " bin_count_plus_one"] | None = None,
    bin_count: int = 10,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
    factor_label: str | None = None,
    category_labels: Sequence[str] | None = None,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot equal-participant-weighted recall rates using common measurement bins.

    Recall classification and counts follow ``cat_recall_by_factor``: response
    codes identify positions, without expansion across repeated presentations.
    Each curve averages the available participant rates in each bin.

    ``labels`` names datasets; ``category_labels`` names categories. When both
    vary, curve labels combine the two. The x-axis defaults to ``factor_field``;
    ``factor_label`` can supply descriptive text and units. Other styling and
    confidence-interval arguments follow the shared plotting functions.

    Automatic edges come from finite, eligible study positions in the selected
    trials of all supplied datasets, regardless of category. Explicit edges
    must be finite and strictly increasing. Empty trial selections are skipped;
    automatic binning without any eligible reference values raises ValueError.

    Supply nonempty datasets, categories and colors, one mask and label per dataset, and
    one category label per category.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    categories = tuple(category_values)
    if labels is None:
        labels = [
            f"Dataset {i + 1}" if len(datasets) > 1 else ""
            for i in range(len(datasets))
        ]
    if category_labels is None:
        category_labels = [
            str(value) if len(categories) > 1 else "" for value in categories
        ]

    if bin_edges is None:
        bin_edges = _shared_factor_bin_edges(
            datasets, trial_masks, factor_field, valid_field, bin_count
        )
        if not np.all(np.isfinite(np.asarray(bin_edges))):
            raise ValueError(
                "Automatic binning needs at least one finite eligible study value."
            )
    edges = jnp.asarray(bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for data_index, (data, mask) in enumerate(zip(datasets, trial_masks)):
        values = subject_cat_recall_by_factor(
            data,
            mask,
            category_field,
            categories,
            factor_field,
            edges,
            valid_field=valid_field,
        )
        if values.shape[0] == 0:
            continue
        for category_index, category_label in enumerate(category_labels):
            label = " / ".join(
                part for part in (labels[data_index], category_label) if part
            )
            color_index = data_index * len(categories) + category_index
            plot_mean_curve(
                axis,
                centers,
                values[:, category_index, :],
                label,
                color_cycle[color_index % len(color_cycle)],
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
    format_analysis_axes(
        axis,
        factor_field if factor_label is None else factor_label,
        "Recall Rate",
        contrast_name,
    )
    return axis


def _shared_factor_bin_edges(
    datasets: Sequence[RecallDataset],
    trial_masks: Sequence[Bool[Array, " trials"]],
    factor_field: str,
    valid_field: str | None,
    bin_count: int,
) -> Float[Array, " bin_edges"]:
    """Prepare common bins from every selected, valid study position."""
    reference_values, reference_masks = [], []
    for data, trial_mask in zip(datasets, trial_masks):
        measurements = data[factor_field]
        included = study_position_mask(data, valid_field)
        reference_values.append(measurements.reshape(-1))
        reference_masks.append((included & trial_mask[:, None]).reshape(-1))
    return compute_factor_bin_edges(
        jnp.concatenate(reference_values), jnp.concatenate(reference_masks), bin_count
    )
