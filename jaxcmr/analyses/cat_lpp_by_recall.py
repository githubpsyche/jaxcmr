"""Category-filtered LPP analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from matplotlib.axes import Axes

from ..helpers import find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Integer, RecallDataset

__all__ = ["plot_cat_lpp_by_recall", "expand_categories_by_recall"]


def expand_categories_by_recall(
    dataset: RecallDataset,
    category_field: str,
) -> Integer[Array, " trial_count study_positions"]:
    """Returns category labels split by recall outcome.

    Args:
      dataset: Recall dataset providing study event metadata.
      category_field: Key in ``dataset`` with per-item category labels.
    """
    categories = dataset[category_field]
    presentations = dataset["pres_itemnos"]
    recalls = dataset["recalls"]

    recall_hits = jnp.any(
        (recalls[:, :, None] == presentations[:, None, :]) & (recalls[:, :, None] > 0),
        axis=1,
    )
    recall_flags = recall_hits.astype(categories.dtype)
    remapped = categories * 2 - 1 + recall_flags
    return jnp.array(jnp.where(categories > 0, remapped, 0), dtype=jnp.int32)


def plot_cat_lpp_by_recall(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_value: int | Sequence[int],
    lpp_field: str = "LateLPP",
    exclude_ci=False,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Returns Matplotlib ``Axes`` with recall-filtered LPP curves for specified category.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        category_field: Keys providing item categories per study position.
        category_value: Category value to compute the LPPs over.
        lpp_field: Key in ``dataset`` providing LPP values per study position.
        exclude_ci: If ``True``, confidence intervals will not be plotted.
        color_cycle: Colors for plotting each dataset.
        labels: Labels per dataset or category. Assumed per-category if multiple values provided.
        contrast_name: Legend title for contrasts.
        axis: Existing Matplotlib ``Axes`` to plot on.
        confidence_level: Confidence level for the bounds.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * 2

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, _data in enumerate(datasets):
        # Expand category labels by recall outcome
        data = _data.copy()
        if exclude_ci:
            data["subject"] = (data["subject"] * 0) + 1
        data[category_field] = expand_categories_by_recall(data, category_field)

        # Select both recalled and unrecalled versions of the category value
        if type(category_value) is int:
            category_values = [category_value * 2, category_value * 2 - 1]
        elif type(category_value) is list:
            category_values = category_value
        else:
            raise ValueError("category_value must be int or list of int")

        for label_index, _category_value in enumerate(category_values):
            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(
                        cat_lpp_spc,
                        static_argnames=(
                            "category_field",
                            "category_value",
                            "lpp_field",
                        ),
                    ),
                    category_field=category_field,
                    category_value=_category_value,
                    lpp_field=lpp_field,
                )
            )

            color_idx = data_index * len(category_values) + label_index
            color = color_cycle[color_idx % len(color_cycle)]
            plot_data(
                axis,
                jnp.arange(max_list_length, dtype=int) + 1,
                subject_values,
                labels[label_index] if len(category_values) > 1 else labels[data_index],
                color,
                confidence_level=confidence_level,
            )

    set_plot_labels(axis, "Study Position", f"{lpp_field} (uV)", contrast_name)
    return axis
