"""Category-filtered serial position curve analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["cat_spc", "plot_cat_spc"]


def _category_recall_counts(
    recalls: Integer[Array, " recall_events"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recall counts per position restricted to a category."""
    position_counts = jnp.bincount(recalls, length=list_length + 1)[1:]
    return position_counts * (categories == category_value)


def cat_spc(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
) -> Float[Array, " study_positions"]:
    """Returns category-filtered recall rate as a function of study position.

    Args:
        dataset: Recall dataset containing per-item category metadata.
        category_field: Key in ``dataset`` providing item categories per study position.
        category_value: Category value to compute the SPC over.
    """
    recalls = dataset["recalls"]
    categories = dataset[category_field]
    list_length = categories.shape[1]

    recall_counts = vmap(_category_recall_counts, in_axes=(0, 0, None, None))(
        recalls,
        categories,
        category_value,
        list_length,
    )

    numerator = recall_counts.sum(axis=0)
    denominator = jnp.sum(categories == category_value, axis=0)
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


def plot_cat_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Matplotlib ``Axes`` with category-filtered SPC curves.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        category_field: Keys providing item categories per study position.
        category_value: Category values to compute the SPC over.
        color_cycle: Colors for plotting each dataset.
        labels: Legend labels for each dataset.
        contrast_name: Legend title for contrasts.
        axis: Existing Matplotlib ``Axes`` to plot on.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks] * len(datasets)

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):

        for category_value in category_values:

            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(cat_spc, static_argnames=("category_field", "category_value")),
                    category_field=category_field,
                    category_value=category_value,
                )
            )


            color = color_cycle.pop(0)
            plot_data(
                axis,
                jnp.arange(max_list_length, dtype=int) + 1,
                subject_values,
                labels[data_index],
                color,
            )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
