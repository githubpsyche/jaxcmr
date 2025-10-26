"""Category-filtered LPP by recall outcome analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = [
    "trial_category_lpp_recall_sums",
    "category_lpp_recall_means",
    "cat_lpp_recall",
    "plot_cat_lpp_recall",
]


def trial_category_lpp_recall_sums(
    recalls: Integer[Array, " recall_positions"],
    lpp: Float[Array, " study_positions"],
    categories: Integer[Array, " study_positions"],
    category_value: int,
    list_length: int,
) -> tuple[Float[Array, " recall_outcome"], Float[Array, " recall_outcome"]]:
    """Returns LPP sums and counts per recall outcome for one trial.

    Args:
        recalls: Recall events for a trial. 1-indexed; 0 for no recall.
        lpp: LPP values per study position for the trial.
        categories: Item category codes per study position.
        category_value: Category value to aggregate over.
        list_length: Length of the study list.
    """
    category_mask = (categories == category_value).astype(lpp.dtype)
    recall_counts = jnp.bincount(recalls, length=list_length + 1)[1:]
    recall_flags = (recall_counts > 0).astype(lpp.dtype)
    recalled_mask = category_mask * recall_flags
    not_recalled_mask = category_mask - recalled_mask

    lpp_sums = jnp.stack(
        [
            (lpp * not_recalled_mask).sum(),
            (lpp * recalled_mask).sum(),
        ]
    )
    counts = jnp.stack(
        [
            not_recalled_mask.sum(),
            recalled_mask.sum(),
        ]
    )
    return lpp_sums, counts


def category_lpp_recall_means(
    recalls: Integer[Array, " trial_count recall_positions"],
    lpp: Float[Array, " trial_count study_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
    list_length: int,
) -> Float[Array, " recall_outcome"]:
    """Returns mean LPP per recall outcome restricted to a category.

    Args:
        recalls: Trial by recall position array of recalled items. 1-indexed; 0 for no recall.
        lpp: Trial by study position array of LPP values.
        categories: Trial by study position array of item categories.
        category_value: Category value to aggregate over.
        list_length: Length of the study list.
    """
    lpp_sums, counts = vmap(
        trial_category_lpp_recall_sums,
        in_axes=(0, 0, 0, None, None),
    )(recalls, lpp, categories, category_value, list_length)

    total_sums = lpp_sums.sum(axis=0)
    total_counts = counts.sum(axis=0)
    return jnp.where(total_counts > 0, total_sums / total_counts, 0.0)


def cat_lpp_recall(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    lpp_field: str = "LateLPP",
) -> Float[Array, " recall_outcome"]:
    """Returns category-filtered mean LPP by recall outcome.

    Args:
        dataset: Recall dataset containing per-item LPP metadata.
        category_field: Key in ``dataset`` providing item categories per study position.
        category_value: Category value to compute means over.
        lpp_field: Key in ``dataset`` providing LPP values per study position.
    """
    recalls = dataset["recalls"]
    lpp = dataset[lpp_field]
    categories = dataset[category_field]
    list_length = categories.shape[1]

    return category_lpp_recall_means(
        recalls,
        lpp,
        categories,
        category_value,
        list_length,
    )


def plot_cat_lpp_recall(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    category_field: str,
    category_values: Sequence[int],
    lpp_field: str = "LateLPP",
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Matplotlib ``Axes`` with category-filtered LPP by recall outcome.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        category_field: Keys providing item categories per study position.
        category_values: Category values to compute the means over.
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
        size = len(category_values) if len(category_values) > 1 else len(datasets)
        labels = [""] * size

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks] * len(datasets)

    x_positions = jnp.array([0.0, 1.0])
    for data_index, data in enumerate(datasets):
        for label_index, category_value in enumerate(category_values):
            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(
                        cat_lpp_recall,
                        static_argnames=(
                            "category_field",
                            "category_value",
                            "lpp_field",
                        ),
                    ),
                    category_field=category_field,
                    category_value=category_value,
                    lpp_field=lpp_field,
                )
            )

            color = color_cycle.pop(0)
            plot_data(
                axis,
                x_positions,
                subject_values,
                labels[label_index] if len(category_values) > 1 else labels[data_index],
                color,
            )

    set_plot_labels(axis, "Recall Outcome", f"{lpp_field} (uV)", contrast_name)
    return axis
