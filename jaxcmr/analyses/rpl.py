"""Recall probability as a function of repeated-item spacing.

``item_lag_counts`` counts one item's opportunity and recall. Dataset operations
pool these counts across lists; participant operations retain separate rows.
Spacing is the number of intervening items between the first two presentations.
Once-presented items occupy a separate leading column.

``recall_probability_by_lag`` reports individual spacings with an overflow
bucket at its configured maximum. ``binned_recall_probability_by_lag`` instead
accepts explicit inclusive ranges and divides their pooled recalled-item counts
by their pooled opportunities. It does not average per-spacing probabilities.
Plots give each participant equal weight. Slope comparisons use explicit spacing
coordinates and omit the once-presented column.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._study_positions import study_position_mask
from .statistics import compare_paired, compare_to_reference

__all__ = [
    "binned_recall_probability_by_lag",
    "infer_max_lag",
    "item_lag_counts",
    "plot_full_rpl",
    "plot_rpl",
    "recall_probability_by_lag",
    "run_rpl_slope_analysis",
    "subject_binned_rpl",
    "subject_full_rpl",
    "test_rpl_slope",
    "test_rpl_slope_vs_comparison",
]


def infer_max_lag(
    presentations: Integer[Array, " trials study_events"],
    list_length: int,
) -> int:
    """Return the largest spacing between first and second presentations.

    Parameters
    ----------
    presentations : array of shape (trials, study_events)
        Positive within-list item identities; zero pads unused positions.
    list_length : int
        Number of stored study positions to inspect.

    Returns
    -------
    int
        Largest number of intervening items, or zero if there are no repeats.
        This host-side operation is used to choose a static array size.
    """
    maximum = 0
    for presentation in np.asarray(presentations)[:, :list_length]:
        for item in np.unique(presentation[presentation > 0]):
            positions = np.flatnonzero(presentation == item)
            if positions.size >= 2:
                maximum = max(maximum, int(positions[1] - positions[0] - 1))
    return maximum


def item_lag_counts(
    target_item: Int_,
    recalls: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    max_lag: int,
    n_bins: int,
) -> tuple[Bool[Array, " lag_bins"], Bool[Array, " lag_bins"]]:
    """One-hot presented/recalled vectors for an item's lag bin.

    Parameters
    ----------
    target_item : Int_
        Item ID to tabulate.
    recalls : Integer[Array, " recall_events"]
        1-indexed recalled positions; 0 for no recall.
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    max_lag : int
        Largest explicit lag bucket.
    n_bins : int
        Total number of bins (``max_lag + 2``).

    Returns
    -------
    tuple[Bool[Array, " lag_bins"], Bool[Array, " lag_bins"]]
        Presented and recalled one-hot vectors.

    """
    first_pos, second_pos = item_to_study_positions(target_item, presentation, size=2)
    lag = jnp.maximum(second_pos - first_pos, 0)
    bin_index = jnp.clip(lag, 0, max_lag + 1)

    presented = (jnp.arange(n_bins) == bin_index) & (first_pos > 0)
    response_indices = jnp.clip(recalls - 1, 0, presentation.size - 1)
    valid = (recalls > 0) & (recalls <= presentation.size)
    recalled = presented & jnp.any(
        valid & (presentation[response_indices] == target_item)
    )

    return presented, recalled


def _trial_lag_counts(
    recalls: Integer[Array, " recall_events"],
    presentations: Integer[Array, " study_events"],
    all_items: Integer[Array, " items"],
    max_lag: int,
    n_bins: int,
) -> tuple[Float[Array, " lag_bins"], Float[Array, " lag_bins"]]:
    """Aggregate presented and recalled one-hot vectors for one trial."""
    presented, recalled = vmap(item_lag_counts, in_axes=(0, None, None, None, None))(
        all_items, recalls, presentations, max_lag, n_bins
    )
    return presented.sum(0), recalled.sum(0)


def _dataset_lag_counts(
    dataset: RecallDataset,
    max_lag: int,
) -> tuple[Integer[Array, " lag_bins"], Integer[Array, " lag_bins"]]:
    """Pool eligible item opportunities and recalled items by spacing."""
    presentations = jnp.where(study_position_mask(dataset), dataset["pres_itemnos"], 0)
    items = vmap(lambda row: jnp.unique(row, size=row.size, fill_value=0))(
        presentations
    )
    actual, recalled = vmap(_trial_lag_counts, in_axes=(0, 0, 0, None, None))(
        dataset["recalls"],
        presentations,
        items,
        max_lag,
        max_lag + 2,
    )
    return actual.sum(axis=0), recalled.sum(axis=0)


def recall_probability_by_lag(
    dataset: RecallDataset,
    max_lag: int = 8,
) -> Float[Array, " lag_bins"]:
    """Recall probability by repetition lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    max_lag : int
        Largest explicit lag bucket.

    Returns
    -------
    Float[Array, " lag_bins"]
        Recall probability per lag bin.

    """
    presented, recalled = _dataset_lag_counts(dataset, max_lag)
    return recalled / presented


def binned_recall_probability_by_lag(
    dataset: RecallDataset,
    spacing_bins: Sequence[tuple[int, int]],
) -> Float[Array, " spacing_bins_plus_singletons"]:
    """Pool item counts within explicit spacing ranges, then divide.

    Parameters
    ----------
    dataset : RecallDataset
        Trials to pool, normally belonging to one participant.
    spacing_bins : sequence of (lower, upper) pairs
        Inclusive ranges of intervening items, in output order. For example,
        ``((0, 0), (1, 2), (3, 5), (6, 8))``. Supply disjoint, nonnegative
        ranges; spacings outside them are excluded. Bins must be static under JIT.

    Returns
    -------
    array of shape (len(spacing_bins) + 1,)
        Once-presented recall probability followed by each range's recalled
        item count divided by eligible item count. Empty groups are NaN.
        The first two presentations determine spacing for items studied more
        than twice. There is no equal-spacing averaging or overflow bin here.
    """
    presented, recalled = _dataset_lag_counts(
        dataset, max(dataset["pres_itemnos"].shape[1] - 2, 0)
    )
    spacings = jnp.arange(presented.size - 1)
    probabilities = [recalled[0] / presented[0]]
    for lower, upper in spacing_bins:
        selected = (spacings >= lower) & (spacings <= upper)
        probabilities.append(
            jnp.sum(jnp.where(selected, recalled[1:], 0))
            / jnp.sum(jnp.where(selected, presented[1:], 0))
        )
    return jnp.stack(probabilities)


def plot_full_rpl(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot full-resolution repetition-lag curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
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

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with the RPL plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)
    max_lag = infer_max_lag(
        datasets[0]["pres_itemnos"], datasets[0]["pres_itemnos"].shape[1]
    )
    xticklabels = ["N/A"] + [f"{i}" for i in range(max_lag + 1)]
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(recall_probability_by_lag, static_argnames=("max_lag",)),
                max_lag,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            jnp.arange(len(xticklabels)),
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    axis.set_xticks(range(len(xticklabels)))
    axis.set_xticklabels(xticklabels)
    format_analysis_axes(axis, "Lag", "Recall Rate", contrast_name)
    return axis


def plot_rpl(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    spacing_bins: Sequence[tuple[int, int]],
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot binned repetition-lag curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
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

    spacing_bins : sequence of (lower, upper) pairs
        Inclusive ranges of intervening items. Pool counts within each
        participant before averaging participant probabilities.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with the binned RPL plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)
    xticklabels = ["N/A"] + [
        str(low) if low == high else f"{low}-{high}" for low, high in spacing_bins
    ]
    for data_index, data in enumerate(datasets):
        subject_values = subject_binned_rpl(data, trial_masks[data_index], spacing_bins)
        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            jnp.arange(len(xticklabels)),
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    axis.set_xticks(range(len(xticklabels)))
    axis.set_xticklabels(xticklabels)
    format_analysis_axes(axis, "Lag", "Recall Rate", contrast_name)
    return axis


def _resolve_max_lag(
    dataset: RecallDataset,
    max_lag: int | None,
) -> int:
    """Return the explicit lag bucket limit.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset used to infer the limit when ``max_lag`` is None.
    max_lag : int or None
        Explicit lag bucket limit if provided.

    Returns
    -------
    int
        Resolved lag bucket limit.

    """
    if max_lag is not None:
        return max_lag
    return infer_max_lag(dataset["pres_itemnos"], dataset["pres_itemnos"].shape[1])


def subject_full_rpl(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int | None = None,
) -> Float[np.ndarray, "subjects lags"]:
    """Subject-level recall probability by lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    max_lag : int or None
        Largest explicit lag bucket.

    Returns
    -------
    np.ndarray
        Shape [n_subjects, max_lag + 2].

    """
    max_lag = _resolve_max_lag(dataset, max_lag)
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(recall_probability_by_lag, static_argnames=("max_lag",)),
        max_lag,
    )
    return np.vstack(subject_values)


def subject_binned_rpl(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    spacing_bins: Sequence[tuple[int, int]],
) -> Float[np.ndarray, " subjects spacing_bins_plus_singletons"]:
    """Return pooled-bin probabilities for each selected participant.

    Parameters
    ----------
    dataset : RecallDataset
        Study lists and responses with participant IDs.
    trial_mask : boolean array of shape (trial_count,)
        Selected trials.
    spacing_bins : sequence of (lower, upper) pairs
        Inclusive spacing ranges as in ``binned_recall_probability_by_lag``.

    Returns
    -------
    numpy.ndarray
        Participants in sorted ID order; columns are once-presented items,
        then the supplied spacing ranges. Average rows to weight participants
        equally, rather than pooling participants' item counts.
    """
    bins = tuple(tuple(bounds) for bounds in spacing_bins)
    values = apply_by_subject(
        dataset,
        trial_mask,
        jit(binned_recall_probability_by_lag, static_argnames=("spacing_bins",)),
        bins,
    )
    return np.vstack(values) if values else np.empty((0, len(bins) + 1))


def _fit_subject_slopes(
    subject_values: Float[np.ndarray, "subjects lags"],
    lag_values: Float[np.ndarray, " lags"],
) -> Float[np.ndarray, " subjects"]:
    """Per-subject slope estimates for recall probability by lag.

    Parameters
    ----------
    subject_values : np.ndarray
        Subject-level recall probabilities for each lag.
    lag_values : np.ndarray
        Lag values aligned to columns in ``subject_values``.

    Returns
    -------
    np.ndarray
        Shape ``[n_subjects]``; NaN where insufficient data.

    """
    slopes = np.full(subject_values.shape[0], np.nan)
    for subject_index, values in enumerate(subject_values):
        valid = np.isfinite(values)
        if np.unique(lag_values[valid]).size < 2:
            continue
        slopes[subject_index] = np.polyfit(lag_values[valid], values[valid], 1)[0]
    return slopes


def test_rpl_slope(
    subject_rpl: Float[np.ndarray, " subjects spacing_bins_plus_singletons"],
    spacing_values: Float[np.ndarray, " spacing_bins"],
) -> pd.DataFrame:
    """Compare participant spacing slopes with zero.

    Parameters
    ----------
    subject_rpl : array of shape (subjects, spacing_bins + 1)
        Participant probabilities, with once-presented items in the first column.
    spacing_values : array of shape (spacing_bins,)
        Explicit coordinates for the remaining columns, such as bin midpoints.

    Returns
    -------
    pandas.DataFrame
        One ``slope`` row with the columns of ``compare_to_reference``. Each
        participant needs two finite probabilities at distinct coordinates.
    """
    slopes = _fit_subject_slopes(
        np.asarray(subject_rpl)[:, 1:], np.asarray(spacing_values)
    )
    return compare_to_reference(slopes, labels=["slope"])


def test_rpl_slope_vs_comparison(
    observed_rpl: Float[np.ndarray, " subjects spacing_bins_plus_singletons"],
    comparison_rpl: Float[np.ndarray, " subjects spacing_bins_plus_singletons"],
    spacing_values: Float[np.ndarray, " spacing_bins"],
) -> pd.DataFrame:
    """Compare spacing slopes from aligned participant probabilities.

    Parameters
    ----------
    observed_rpl, comparison_rpl : array of shape (subjects, spacing_bins + 1)
        Matching participant rows and spacing columns, including the leading
        once-presented column that is excluded from each slope.
    spacing_values : array of shape (spacing_bins,)
        Explicit spacing coordinates shared by both calculations.

    Returns
    -------
    pandas.DataFrame
        One ``slope`` row, using participants with a defined slope on both sides.
    """
    coordinates = np.asarray(spacing_values)
    observed = _fit_subject_slopes(np.asarray(observed_rpl)[:, 1:], coordinates)
    comparison = _fit_subject_slopes(np.asarray(comparison_rpl)[:, 1:], coordinates)
    return compare_paired(observed, comparison, labels=["slope"])


def run_rpl_slope_analysis(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    comparison_dataset: RecallDataset,
    comparison_mask: Bool[Array, " comparison_trials"],
    spacing_values: Float[np.ndarray, " spacing_bins"],
    *,
    spacing_bins: Sequence[tuple[int, int]] | None = None,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Calculate and compare participant spacing slopes for two datasets.

    Parameters
    ----------
    dataset, comparison_dataset : RecallDataset
        Datasets with the same selected participant identities.
    trial_mask, comparison_mask : boolean arrays
        Selected trials in their respective datasets.
    spacing_values : array of shape (spacing_bins,)
        Explicit coordinates, excluding once-presented items.
    spacing_bins : sequence of (lower, upper) pairs, optional
        Use pooled bins when supplied; otherwise use individual spacings.
    max_lag : int, optional
        Full-RPL maximum when bins are omitted. Inferred from the first dataset
        when absent, then used for both datasets.

    Returns
    -------
    pandas.DataFrame
        Rows ``observed``, ``comparison`` and ``difference``. The first two test
        slopes against zero; the last compares paired slopes.

    Raises
    ------
    ValueError
        The selected participant identities differ.
    """
    observed_ids = np.unique(
        np.asarray(dataset["subject"]).reshape(-1)[np.asarray(trial_mask)]
    )
    comparison_ids = np.unique(
        np.asarray(comparison_dataset["subject"]).reshape(-1)[
            np.asarray(comparison_mask)
        ]
    )
    if not np.array_equal(observed_ids, comparison_ids):
        raise ValueError(
            "Spacing-slope comparisons require the same selected participants."
        )
    if spacing_bins is None:
        limit = _resolve_max_lag(dataset, max_lag)
        observed = subject_full_rpl(dataset, trial_mask, limit)
        comparison = subject_full_rpl(comparison_dataset, comparison_mask, limit)
    else:
        observed = subject_binned_rpl(dataset, trial_mask, spacing_bins)
        comparison = subject_binned_rpl(
            comparison_dataset, comparison_mask, spacing_bins
        )
    results = [
        test_rpl_slope(observed, spacing_values),
        test_rpl_slope(comparison, spacing_values),
        test_rpl_slope_vs_comparison(observed, comparison, spacing_values),
    ]
    return pd.concat(results, ignore_index=True).set_axis(
        pd.Index(["observed", "comparison", "difference"], name="comparison")
    )
