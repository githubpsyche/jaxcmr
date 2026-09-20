"""LPP convenience wrappers for category-filtered numeric factor means.

Use ``cat_factor_spc`` for other study measurements. These entry points retain
the historical LPP field defaults, units, and plot-label convention.
"""

from collections.abc import Sequence

from matplotlib.axes import Axes

from ..typing import Array, Bool, Float, Integer, RecallDataset
from .cat_factor_spc import (
    cat_factor_spc,
    category_factor_values,
    plot_cat_factor_spc,
)

__all__ = ["cat_lpp_spc", "category_lpp_values", "plot_cat_lpp_spc"]


def category_lpp_values(
    lpp: Float[Array, " trial_count study_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
) -> Float[Array, " study_positions"]:
    """Return finite mean LPP per position within a category; empty cells are NaN."""
    return category_factor_values(lpp, categories, category_value)


def cat_lpp_spc(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    lpp_field: str = "EarlyLPP",
    *,
    valid_field: str | None = None,
) -> Float[Array, " study_positions"]:
    """Return pooled LPP means by study position for a category.

    Missing measurements, padding, and positions beyond listLength are excluded
    from both sums and counts. ``valid_field`` optionally supplies finite,
    positive inclusion flags. Field names must be static under JIT.
    """
    return cat_factor_spc(
        dataset, category_field, category_value, lpp_field, valid_field=valid_field
    )


def plot_cat_lpp_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    lpp_field: str = "LateLPP",
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    valid_field: str | None = None,
    draw_ci: bool = True,
) -> Axes:
    """Plot equal-subject-weighted LPP means in microvolts.

    As in the original interface, ``labels`` names categories when several
    categories are supplied, otherwise datasets. The generic plot offers
    separate dataset/category labels. ``draw_ci`` only controls display.
    """
    return plot_cat_factor_spc(
        datasets, trial_masks, category_field, category_values, lpp_field,
        color_cycle=color_cycle,
        labels=labels if len(category_values) <= 1 else None,
        contrast_name=contrast_name, axis=axis, confidence_level=confidence_level,
        valid_field=valid_field,
        category_labels=labels if len(category_values) > 1 else None,
        factor_label=f"{lpp_field} (uV)", draw_ci=draw_ci,
    )
