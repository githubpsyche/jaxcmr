"""LPP convenience wrappers for factor means split by recall status.

The generic ``cat_factor_by_recall`` module keeps category and recall status
separate. The legacy category-encoding helper remains available for callers
that use its integer codes.
"""

from collections.abc import Sequence
from numbers import Integral

from jax import numpy as jnp
from matplotlib.axes import Axes

from ..typing import Array, Bool, Integer, RecallDataset
from ._study import recall_status, study_field, study_validity
from .cat_factor_by_recall import subject_cat_factor_by_recall
from .cat_factor_spc import _plot_factor_curves

__all__ = ["expand_categories_by_recall", "plot_cat_lpp_by_recall"]


def expand_categories_by_recall(
    dataset: RecallDataset,
    category_field: str,
    *,
    expand_repetitions: bool = False,
) -> Integer[Array, " trial_count study_positions"]:
    """Encode positive category ``c`` as ``2*c`` recalled or ``2*c-1`` otherwise.

    Nonpositive categories and invalid study positions map to zero. Recall
    codes identify study positions, not item IDs. ``expand_repetitions=True``
    also marks valid occurrences of the same item as a directly recalled
    position. Field names and the expansion choice must be static under JIT.
    """
    categories = study_field(dataset, category_field)
    hits = recall_status(dataset, expand_repetitions=expand_repetitions)
    remapped = categories * 2 - 1 + hits.astype(categories.dtype)
    valid = study_validity(dataset) & (categories > 0)
    return jnp.asarray(jnp.where(valid, remapped, 0), dtype=jnp.int32)


def plot_cat_lpp_by_recall(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_value: int | Sequence[int],
    lpp_field: str = "LateLPP",
    exclude_ci: bool = False,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
    expand_repetitions: bool = False,
) -> Axes:
    """Plot equal-subject-weighted LPP means in microvolts by recall status.

    An integer category selects its recalled then not-recalled curves. For
    compatibility, a sequence selects already-expanded codes: even ``2*c``
    selects category ``c`` recalled; odd ``2*c-1`` selects it not recalled.
    These legacy codes require positive categories. The generic plot supports
    any integer category with a separate recall-status axis.

    ``labels`` names curves when several are selected, otherwise datasets.
    ``exclude_ci`` hides intervals without changing subject averaging. Missing
    measurements and invalid study positions are excluded; ``valid_field``
    can add quality flags. Input datasets are never modified.
    """
    if isinstance(category_value, (int, Integral)):
        codes = (int(category_value) * 2, int(category_value) * 2 - 1)
    else:
        codes = tuple(category_value)
    if not codes or any(code <= 0 for code in codes):
        raise ValueError("Select a positive category or positive expanded category codes.")
    categories = tuple(dict.fromkeys((code + 1) // 2 for code in codes))
    selections = [(categories.index((code + 1) // 2), code % 2) for code in codes]
    curve_labels = [
        " / ".join(part for part in (
            str(categories[index]) if len(categories) > 1 else "",
            "Recalled" if status == 0 else "Not recalled",
        ) if part)
        for index, status in selections
    ]
    if labels is not None and len(codes) > 1:
        if len(labels) != len(codes):
            raise ValueError("labels must contain one label per selected curve.")
        curve_labels = list(labels)
    elif len(codes) == 1:
        curve_labels = [""]

    def summarize(data: RecallDataset, mask: Array) -> Array:
        values = subject_cat_factor_by_recall(
            data, mask, category_field, categories, lpp_field,
            valid_field=valid_field, expand_repetitions=expand_repetitions,
        )
        return jnp.stack([values[:, index, status, :] for index, status in selections], axis=1)

    return _plot_factor_curves(
        datasets, trial_masks, summarize, curve_labels, f"{lpp_field} (uV)",
        color_cycle=color_cycle, labels=labels if len(codes) == 1 else None,
        contrast_name=contrast_name, axis=axis, confidence_level=confidence_level,
        draw_ci=not exclude_ci,
    )
