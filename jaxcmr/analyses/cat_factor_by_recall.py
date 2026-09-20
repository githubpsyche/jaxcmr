"""Numeric factor means by category, recall status, and study position.

The status axis is always [recalled, not recalled]; category labels are never
rewritten. Positional recall is the default. Expanding a recalled item to all
its study occurrences is an explicit, optional choice.
"""

from collections.abc import Sequence

from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, RecallDataset
from ._study import factor_inputs, recall_status, validate_trial_mask
from .cat_factor_spc import _plot_factor_curves, category_factor_values

__all__ = [
    "cat_factor_by_recall",
    "plot_cat_factor_by_recall",
    "subject_cat_factor_by_recall",
]


def cat_factor_by_recall(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    factor_field: str,
    *,
    valid_field: str | None = None,
    expand_repetitions: bool = False,
) -> Float[Array, "2 study_positions"]:
    """Return pooled factor means, with rows [recalled, not recalled].

    Only finite measurements at valid study positions contribute. Recall codes
    are one-indexed study positions; duplicates, padding, and invalid codes do
    not add hits. ``expand_repetitions=True`` marks all valid occurrences of an
    item when any of its positions was recalled. The choice does not depend on
    missing factor measurements. Empty category/status/position cells are NaN.

    Field names and ``expand_repetitions`` must be static under JIT. Use the
    subject helper below when participant means, rather than pooled means,
    are the desired summary.
    """
    factor, categories, valid = factor_inputs(
        dataset, category_field, factor_field, valid_field
    )
    recalled = recall_status(dataset, expand_repetitions=expand_repetitions)
    return jnp.stack([
        category_factor_values(factor, categories, category_value, valid=valid & recalled),
        category_factor_values(factor, categories, category_value, valid=valid & ~recalled),
    ])


def subject_cat_factor_by_recall(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trials"],
    category_field: str,
    category_values: Sequence[int],
    factor_field: str,
    *,
    valid_field: str | None = None,
    expand_repetitions: bool = False,
) -> Float[Array, " subjects categories 2 study_positions"]:
    """Return subject-by-category-by-status-by-position factor means.

    Subject IDs are sorted; category order follows the caller; statuses are
    [recalled, not recalled]. Subjects without selected trials are omitted.
    Empty selections return a zero-length subject axis. Grouping runs outside
    JIT; each subject calculation is compiled. Averaging the subject axis with
    ``jnp.nanmean`` weights available subjects equally in each cell.
    """
    categories = tuple(category_values)
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    validate_trial_mask(dataset, trial_mask)

    @jit
    def summarize(data: RecallDataset) -> Array:
        return jnp.stack([
            cat_factor_by_recall(
                data, category_field, category, factor_field,
                valid_field=valid_field, expand_repetitions=expand_repetitions,
            )
            for category in categories
        ])

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
    expand_repetitions: bool = False,
    category_labels: Sequence[str] | None = None,
    recall_labels: Sequence[str] = ("Recalled", "Not recalled"),
    factor_label: str | None = None,
    draw_ci: bool = True,
) -> Axes:
    """Plot subject-mean factor curves, separating category and recall status.

    ``labels`` names datasets. Category and recall labels name their respective
    axes, and combined labels identify each plotted curve. Hiding confidence
    intervals never changes the averaging or input dataset. Factor units can
    be supplied through ``factor_label``.
    """
    categories = tuple(category_values)
    if not categories:
        raise ValueError("category_values must contain at least one category.")
    if category_labels is None:
        category_labels = [str(value) if len(categories) > 1 else ""
                           for value in categories]
    if len(category_labels) != len(categories) or len(recall_labels) != 2:
        raise ValueError("Supply one label per category and two recall-status labels.")
    curve_labels = [
        " / ".join(part for part in (category_label, recalled_label) if part)
        for category_label in category_labels for recalled_label in recall_labels
    ]

    def summarize(data: RecallDataset, mask: Array) -> Array:
        values = subject_cat_factor_by_recall(
            data, mask, category_field, categories, factor_field,
            valid_field=valid_field, expand_repetitions=expand_repetitions,
        )
        return values.reshape(values.shape[0], len(categories) * 2, values.shape[-1])

    return _plot_factor_curves(
        datasets, trial_masks, summarize, curve_labels,
        factor_field if factor_label is None else factor_label,
        color_cycle=color_cycle, labels=labels, contrast_name=contrast_name,
        axis=axis, confidence_level=confidence_level, draw_ci=draw_ci,
    )
