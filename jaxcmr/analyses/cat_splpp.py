"""Category-filtered LPP analyses."""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["category_lpp_values", "cat_splpp", "plot_cat_splpp"]


def category_lpp_values(
    lpp: Float[Array, " trial_count study_positions"],
    categories: Integer[Array, " trial_count study_positions"],
    category_value: int,
) -> Float[Array, " study_positions"]:
    """Returns mean LPP per position restricted to a category."""
    matches = categories == category_value
    numerator = jnp.where(matches, lpp, 0.0).sum(axis=0)
    denominator = matches.sum(axis=0)
    return jnp.where(denominator > 0, numerator / denominator, 0.0)


def cat_splpp(
    dataset: RecallDataset,
    category_field: str,
    category_value: int,
    lpp_field: str = "EarlyLPP",
) -> Float[Array, " study_positions"]:
    """Returns category-filtered mean LPP as a function of study position.

    Args:
        dataset: Recall dataset containing per-item LPP metadata.
        category_field: Key in ``dataset`` providing item categories per study position.
        category_value: Category value to compute the LPP curve over.
        lpp_field: Key in ``dataset`` providing LPP values per study position.
    """
    lpp = dataset[lpp_field]
    categories = dataset[category_field]
    return category_lpp_values(lpp, categories, category_value)


def plot_cat_splpp(
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
    """Returns Matplotlib ``Axes`` with category-filtered LPP curves.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        category_field: Keys providing item categories per study position.
        category_values: Category values to compute the LPP over.
        lpp_field: Key in ``dataset`` providing LPP values per study position.
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
        for label_index, category_value in enumerate(category_values):
            subject_values = jnp.vstack(
                apply_by_subject(
                    data,
                    trial_masks[data_index],
                    jit(
                        cat_splpp,
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
                jnp.arange(max_list_length, dtype=int) + 1,
                subject_values,
                labels[label_index],
                color,
            )

    set_plot_labels(axis, "Study Position", f"{lpp_field} (uV)", contrast_name)
    return axis
