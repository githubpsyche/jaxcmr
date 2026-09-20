"""Category-filtered numeric factor means by study position.

Numerical means pool eligible observations across trials. Subject summaries
keep participants separate, and plots average those participant means. Missing
measurements, presentation padding, and optional quality failures are excluded
from both the sums and observation counts. Positions with no data return NaN.
"""

from collections.abc import Callable, Sequence

from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, Real, RecallDataset
from ._study import factor_inputs, validate_trial_mask

__all__ = [
    "cat_factor_spc",
    "category_factor_values",
    "factor_sums_and_counts",
    "plot_cat_factor_spc",
    "subject_cat_factor_spc",
]


def factor_sums_and_counts(
    values: Real[Array, " trials study_positions"],
    include: Bool[Array, " trials study_positions"],
) -> tuple[Float[Array, " study_positions"], Integer[Array, " study_positions"]]:
    """Sum finite included observations and count them at each study position.

    Integer factors are promoted to floating point. Returns zeros for empty
    positions; callers should report an undefined mean when the count is zero.
    Both arrays must have the same two-dimensional shape. Supports JIT.
    """
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    include = jnp.asarray(include, dtype=bool)
    if values.ndim != 2 or include.shape != values.shape:
        raise ValueError("values and include must have matching trial-by-position shapes.")
    valid = include & jnp.isfinite(values)
    return jnp.sum(jnp.where(valid, values, 0), axis=0), jnp.sum(valid, axis=0)


def category_factor_values(
    factor: Real[Array, " trials study_positions"],
    categories: Integer[Array, " trials study_positions"],
    category_value: int,
    *,
    valid: Bool[Array, " trials study_positions"] | None = None,
) -> Float[Array, " study_positions"]:
    """Return finite factor means for one category, optionally restricting rows/cells."""
    if factor.ndim != 2 or categories.shape != factor.shape:
        raise ValueError("factor and categories must have matching trial-by-position shapes.")
    include = categories == category_value
    if valid is not None:
        if valid.shape != factor.shape:
            raise ValueError("valid must match the factor shape.")
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
    """Pool numeric factor measurements by study position within a category.

    Field names select study-position metadata. ``valid_field`` optionally
    supplies finite, positive inclusion flags. Padding and positions past each
    trial's listLength are always excluded. JIT requires field names static;
    the category value may vary dynamically. This analysis does not use recalls.
    """
    factor, categories, valid = factor_inputs(
        dataset, category_field, factor_field, valid_field
    )
    return category_factor_values(factor, categories, category_value, valid=valid)


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
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    validate_trial_mask(dataset, trial_mask)

    @jit
    def summarize(data: RecallDataset) -> Array:
        return jnp.stack([
            cat_factor_spc(
                data, category_field, category, factor_field, valid_field=valid_field
            )
            for category in categories
        ])

    results = apply_by_subject(dataset, trial_mask, summarize)
    if not results:
        return jnp.empty((0, len(categories), dataset["pres_itemnos"].shape[1]))
    return jnp.stack(results)


def _plot_factor_curves(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trials"]] | Bool[Array, " trials"],
    summarize: Callable[[RecallDataset, Array], Array],
    curve_labels: Sequence[str],
    factor_label: str,
    *,
    color_cycle: list[str] | None,
    labels: Sequence[str] | None,
    contrast_name: str | None,
    axis: Axes | None,
    confidence_level: float,
    draw_ci: bool,
) -> Axes:
    """Draw subject-by-curve-by-position summaries with shared display conventions."""
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if not datasets or len(datasets) != len(trial_masks):
        raise ValueError("Supply one trial mask per dataset and at least one dataset.")
    if not curve_labels:
        raise ValueError("Supply at least one curve.")
    if not color_cycle:
        raise ValueError("color_cycle must contain at least one color.")
    if labels is None:
        labels = [f"Dataset {i + 1}" if len(datasets) > 1 else ""
                  for i in range(len(datasets))]
    if len(labels) != len(datasets):
        raise ValueError("labels must contain one label per dataset.")
    for data_index, (data, mask) in enumerate(zip(datasets, trial_masks, strict=True)):
        values = summarize(data, mask)
        if values.shape[0] == 0:
            continue
        positions = jnp.arange(1, values.shape[-1] + 1)
        for curve_index, curve_label in enumerate(curve_labels):
            label = " / ".join(part for part in (labels[data_index], curve_label) if part)
            color_index = data_index * len(curve_labels) + curve_index
            plot_data(
                axis, positions, values[:, curve_index, :], label,
                color_cycle[color_index % len(color_cycle)],
                draw_ci=draw_ci, confidence_level=confidence_level,
            )
    set_plot_labels(axis, "Study Position", factor_label, contrast_name)
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
) -> Axes:
    """Plot equal-subject-weighted factor means by study position and category.

    ``labels`` names datasets; ``category_labels`` names categories. Axis text
    defaults to ``factor_field``; ``factor_label`` can add a description/units.
    ``draw_ci`` controls interval display without changing the subject means.
    """
    categories = tuple(category_values)
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    if category_labels is None:
        category_labels = [str(value) if len(categories) > 1 else ""
                           for value in categories]
    if len(category_labels) != len(categories):
        raise ValueError("category_labels must contain one label per category.")

    def summarize(data: RecallDataset, mask: Array) -> Array:
        return subject_cat_factor_spc(
            data, mask, category_field, categories, factor_field, valid_field=valid_field
        )

    return _plot_factor_curves(
        datasets, trial_masks, summarize, category_labels,
        factor_field if factor_label is None else factor_label,
        color_cycle=color_cycle, labels=labels, contrast_name=contrast_name,
        axis=axis, confidence_level=confidence_level, draw_ci=draw_ci,
    )
