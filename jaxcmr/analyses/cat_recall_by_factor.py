"""Category-specific recall probability binned by a numeric study factor.

Choose bin edges once from reference values, then reuse them for each subject,
category, dataset, and simulation. The numerical analysis pools eligible study
positions; the plotting function averages subject-level rates instead.

Dataset wrappers interpret recalls as one-indexed study positions (zero pads).
They do not expand a recalled item to its other study occurrences. Callers with
other recall conventions can supply their own Boolean indicators directly to
``binned_recall_counts``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, Real, RecallDataset
from ._study import factor_inputs, recalled_positions, validate_trial_mask

__all__ = [
    "binned_recall_counts",
    "cat_recall_by_factor",
    "category_factor_recall_histogram",
    "compute_factor_bin_edges",
    "plot_cat_recall_by_factor",
    "subject_cat_recall_by_factor",
]


def compute_factor_bin_edges(
    values: Real[Array, "..."],
    valid: Bool[Array, "..."],
    bin_count: int = 10,
) -> Float[Array, " bin_count_plus_one"]:
    """Return equal-width edges from finite, included reference values.

    ``valid`` must match ``values``. No eligible values gives all-NaN edges.
    Constant values get a symmetric interval with half-width
    ``max(0.5, abs(value) * 0.01)`` so the edges remain distinct at large scales.
    This helper supports JIT with a static ``bin_count``. It does not inspect
    recall outcomes or infer which datasets/categories should define the bins.
    """
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer.")
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    valid = jnp.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("values and valid must have matching shapes.")
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


def _as_bin_edges(bin_edges: Array) -> Array:
    edges = jnp.asarray(bin_edges)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional array of at least two edges.")
    return edges


def binned_recall_counts(
    recalled: Bool[Array, "..."],
    values: Real[Array, "..."],
    include: Bool[Array, "..."],
    bin_edges: Real[Array, " bin_count_plus_one"],
) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
    """Return recalled and eligible counts for fixed bins, in that order.

    Inputs share a shape, with one Boolean recall indicator per observation.
    Missing/nonfinite and out-of-range values contribute to neither count.
    Bins are left-closed/right-open, except the final bin includes its right
    edge. Empty bins have zero counts; rates derived from them are undefined.

    Edges must be finite and strictly increasing. Invalid edge values yield
    zero counts in every bin, allowing undefined rates to propagate under JIT.
    Malformed edge/input shapes raise ValueError, including during tracing.
    """
    values = jnp.asarray(values)
    recalled = jnp.asarray(recalled, dtype=bool)
    include = jnp.asarray(include, dtype=bool)
    edges = _as_bin_edges(bin_edges)
    if values.shape != recalled.shape or values.shape != include.shape:
        raise ValueError("recalled, values, and include must have matching shapes.")
    bin_count = edges.size - 1
    valid_edges = jnp.all(jnp.isfinite(edges)) & jnp.all(jnp.diff(edges) > 0)
    eligible = (
        include & jnp.isfinite(values) & valid_edges
        & (values >= edges[0]) & (values <= edges[-1])
    )
    indices = jnp.searchsorted(edges, values, side="right") - 1
    indices = jnp.where(values == edges[-1], bin_count - 1, indices)

    # An extra bin absorbs excluded observations without dynamic indexing.
    def count(mask: Array) -> Array:
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

    Repeated responses count once; zero, negative and out-of-range recall codes
    do not mark any position. ``valid`` excludes unusable study positions from
    both numerator and denominator. Empty bins return NaN, never zero.
    """
    if factor.ndim != 2 or categories.shape != factor.shape:
        raise ValueError("factor and categories must have matching trial-by-position shapes.")
    if recalls.ndim != 2 or recalls.shape[0] != factor.shape[0]:
        raise ValueError("recalls must have one row per factor trial.")
    if valid is None:
        valid = jnp.ones(factor.shape, dtype=bool)
    elif valid.shape != factor.shape:
        raise ValueError("valid must match the factor shape.")
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
    """Return pooled recall probability by category and numeric factor bin.

    ``factor_field`` and ``category_field`` select trial-by-study-position
    arrays. ``valid_field``, when supplied, must be finite and positive for a
    position to contribute. Zero-padded presentations and positions beyond
    listLength are always excluded. Recall codes identify study positions.

    Automatic edges use finite, valid study values across *all* categories in
    this dataset. To compare datasets, subjects or simulations, first compute
    reference edges with ``compute_factor_bin_edges`` and pass the same edges
    to every call. This function pools study positions; use
    ``subject_cat_recall_by_factor`` for equal weighting of subjects instead.

    Supports JIT with field names and ``bin_count`` static. The category value
    and explicit bin-edge values may vary dynamically at a fixed shape.
    """
    factor, categories, valid = factor_inputs(
        dataset, category_field, factor_field, valid_field
    )
    if bin_edges is None:
        bin_edges = compute_factor_bin_edges(factor, valid, bin_count)
    return category_factor_recall_histogram(
        dataset["recalls"], factor, categories, category_value, bin_edges, valid=valid
    )


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
    edges = _as_bin_edges(bin_edges)
    categories = tuple(category_values)
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    validate_trial_mask(dataset, trial_mask)

    @jit
    def summarize(data: RecallDataset) -> Array:
        return jnp.stack([
            cat_recall_by_factor(
                data, category_field, category, factor_field, edges,
                valid_field=valid_field,
            )
            for category in categories
        ])

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
) -> Axes:
    """Plot means of subject-level rates, sharing edges across all curves.

    ``labels`` names datasets; ``category_labels`` names categories. When both
    vary, curve labels combine the two. The x-axis defaults to ``factor_field``;
    ``factor_label`` can supply descriptive text and units. Other styling and
    confidence-interval arguments follow the shared plotting functions.

    Automatic edges come from finite, valid study positions in the selected
    trials of all supplied datasets, regardless of category. Explicit edges
    must be finite and strictly increasing. Empty trial selections are skipped;
    automatic binning without any eligible reference values raises ValueError.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    categories = tuple(category_values)
    if not datasets or len(datasets) != len(trial_masks):
        raise ValueError("Supply one trial mask per dataset and at least one dataset.")
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    if not color_cycle:
        raise ValueError("color_cycle must contain at least one color.")
    for data, mask in zip(datasets, trial_masks, strict=True):
        validate_trial_mask(data, mask)
    if labels is None:
        labels = [f"Dataset {i + 1}" if len(datasets) > 1 else ""
                  for i in range(len(datasets))]
    if category_labels is None:
        category_labels = [str(value) if len(categories) > 1 else ""
                           for value in categories]
    if len(labels) != len(datasets) or len(category_labels) != len(categories):
        raise ValueError("labels must match datasets; category_labels must match categories.")

    if bin_edges is None:
        reference_values, reference_masks = [], []
        for data, mask in zip(datasets, trial_masks, strict=True):
            values, _, valid = factor_inputs(
                data, category_field, factor_field, valid_field
            )
            reference_values.append(values.reshape(-1))
            reference_masks.append((valid & mask[:, None]).reshape(-1))
        bin_edges = compute_factor_bin_edges(
            jnp.concatenate(reference_values),
            jnp.concatenate(reference_masks),
            bin_count,
        )
    edges = _as_bin_edges(bin_edges)
    host_edges = np.asarray(edges)
    if not np.all(np.isfinite(host_edges)) or not np.all(np.diff(host_edges) > 0):
        raise ValueError(
            "Bins require finite, strictly increasing edges; automatic binning "
            "needs at least one finite eligible study value."
        )
    centers = 0.5 * (edges[:-1] + edges[1:])
    for data_index, (data, mask) in enumerate(zip(datasets, trial_masks, strict=True)):
        values = subject_cat_recall_by_factor(
            data, mask, category_field, categories, factor_field, edges,
            valid_field=valid_field,
        )
        if values.shape[0] == 0:
            continue
        for category_index, category_label in enumerate(category_labels):
            label = " / ".join(part for part in (labels[data_index], category_label) if part)
            color_index = data_index * len(categories) + category_index
            plot_data(
                axis, centers, values[:, category_index, :], label,
                color_cycle[color_index % len(color_cycle)],
                confidence_level=confidence_level,
            )
    set_plot_labels(
        axis, factor_field if factor_label is None else factor_label,
        "Recall Rate", contrast_name,
    )
    return axis
