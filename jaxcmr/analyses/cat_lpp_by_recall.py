"""Category-filtered LPP analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Integer, RecallDataset
from .cat_lpp_spc import cat_lpp_spc

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
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Matplotlib ``Axes`` with recall-filtered LPP curves for specified category.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        category_field: Keys providing item categories per study position.
        category_value: Category value to compute the LPPs over.
        lpp_field: Key in ``dataset`` providing LPP values per study position.
        color_cycle: Colors for plotting each dataset.
        labels: Labels per dataset or category. Assumed per-category if multiple values provided.
        contrast_name: Legend title for contrasts.
        axis: Existing Matplotlib ``Axes`` to plot on.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * 2

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks] * len(datasets)

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, _data in enumerate(datasets):

        # Expand category labels by recall outcome
        data = _data.copy()
        data[category_field] = expand_categories_by_recall(
            data, category_field
        )

        # Select both recalled and unrecalled versions of the category value
        if type(category_value) is int:
            category_values = [category_value * 2, category_value * 2 - 1]
        else:
            category_values = category_value

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

            color = color_cycle.pop(0)
            plot_data(
                axis,
                jnp.arange(max_list_length, dtype=int) + 1,
                subject_values,
                labels[label_index] if len(category_values) > 1 else labels[data_index],
                color,
            )

    set_plot_labels(axis, "Study Position", f"{lpp_field} (uV)", contrast_name)
    return axis
