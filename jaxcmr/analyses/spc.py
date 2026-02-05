"""Serial position curve analyses.

Utilities to compute and visualize recall rates as a function of study
position.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["fixed_pres_spc", "spc", "plot_spc"]


def fixed_pres_spc(
    recalls: Integer[Array, " trial_count recall_positions"],
    list_length: int,
) -> Float[Array, " study_positions"]:
    """Returns recall rate as a function of study position for uniform lists.

    Args:
        recalls: Trial by recall position array of recalled items. 1-indexed; 0 for no recall.
        list_length: Length of the study list.
    """
    return jnp.bincount(recalls.flatten(), length=list_length + 1)[1:] / len(recalls)


def spc(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Returns recall rate as a function of study position.

    Args:
        dataset: Recall dataset containing at least ``recalls`` and ``pres_itemnos``.
        size: Maximum number of study positions an item can be presented at.
    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]

    expanded_recalls = vmap(
        vmap(all_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, presentations, size)

    counts = jnp.bincount(expanded_recalls.flatten(), length=list_length + 1)[1:]
    return counts / len(expanded_recalls)


def plot_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Returns Matplotlib ``Axes`` with serial position curves for datasets.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks selecting trials in each dataset.
        color_cycle: Colors for plotting each dataset.
        labels: Legend labels for each dataset.
        contrast_name: Legend title for contrasts.
        axis: Existing Matplotlib ``Axes`` to plot on.
        size: Maximum number of study positions an item can be presented at.
        confidence_level: Confidence level for the bounds.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(spc, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            jnp.arange(max_list_length, dtype=int) + 1,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
