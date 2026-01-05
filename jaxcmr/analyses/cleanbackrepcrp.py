"""Plot incoming repetition lag-CRP using reversed recall sequences."""

__all__ = [
    "plot_back_rep_crp",
    "subject_back_rep_crp",
    "test_back_rep_crp_vs_control",
    "test_first_second_bias",
]

from typing import Optional, Sequence

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Integer, RecallDataset
from .backrepcrp import test_back_rep_crp_vs_control, test_first_second_bias
from . import repcrp as repcrp_module


def _reverse_nonzero_recalls(
    recalls: Integer[Array, " trial_count recall_events"],
) -> Integer[Array, " trial_count recall_events"]:
    """Reverse recall sequences while keeping padding at the end.

    Args:
        recalls: Recall sequences with zero padding after termination.
    """

    def reverse_row(
        row: Integer[Array, " recall_events"],
    ) -> Integer[Array, " recall_events"]:
        count = jnp.sum(row > 0)
        idx = jnp.arange(row.size)
        rev_idx = jnp.clip(count - 1 - idx, 0, row.size - 1)
        reversed_part = row[rev_idx]
        return jnp.where(idx < count, reversed_part, 0)

    return vmap(reverse_row)(recalls)


def plot_back_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes with incoming repetition lag-CRP plots for datasets and trial masks.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        max_lag: Maximum lag to plot.
        min_lag: Minimum separation between repeated presentations.
        size: Maximum number of study positions an item can be presented at.
        repetition_index: Specific repetition index to plot, optional.
        color_cycle: List of colors for plotting each dataset.
        labels: Names for each dataset for legend, optional.
        contrast_name: Name of contrast for legend labeling, optional.
        axis: Existing matplotlib Axes to plot on, optional.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    if isinstance(repetition_index, int):
        repetition_indices = [repetition_index]
    else:
        repetition_indices = list(range(size))

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        reversed_data = dict(data)
        reversed_data["recalls"] = _reverse_nonzero_recalls(data["recalls"])
        subject_values = apply_by_subject(
            reversed_data,
            trial_masks[data_index],
            jit(repcrp_module.repcrp, static_argnames=("min_lag", "size")),
            min_lag=min_lag,
            size=size,
        )

        for repetition_index in repetition_indices:
            repetition_subject_values = jnp.vstack(
                [each[repetition_index] for each in subject_values]
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

            color = color_cycle.pop(0)
            plot_data(
                axis,
                lag_interval,
                repetition_subject_values,
                str(repetition_index + 1),
                color,
            )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis


def subject_back_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> np.ndarray:
    """Compute subject-level incoming repetition CRP values.

    Args:
        dataset: Recall dataset.
        trial_mask: Boolean mask selecting trials to include.
        min_lag: Minimum spacing between item repetitions.
        max_lag: Maximum lag to include in output.
        size: Maximum number of presentations per item.

    Returns:
        Array of shape [n_subjects, size, 2*max_lag+1] with CRP values per subject,
        repetition index, and lag.
    """
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    reversed_dataset = dict(dataset)
    reversed_dataset["recalls"] = _reverse_nonzero_recalls(dataset["recalls"])
    subject_values = apply_by_subject(
        reversed_dataset,
        trial_mask,
        jit(repcrp_module.repcrp, static_argnames=("min_lag", "size")),
        min_lag=min_lag,
        size=size,
    )
    return np.stack([s[:, lag_slice] for s in subject_values])
