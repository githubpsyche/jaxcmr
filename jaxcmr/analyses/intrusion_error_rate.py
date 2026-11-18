"""Intrusions at expected serial-recall response positions.

An intrusion is a nonzero response that does not address an item on the study
list. The curve reports the fraction of trials with an intrusion at each
expected response position, including trials with a blank response in that
denominator. It is not conditional on reaching a free-recall output position.
"""

__all__ = [
    "intrusion_error_rate",
    "plot_intrusion_error_rate",
    "trial_intrusion_error_rate",
]

from collections.abc import Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject, find_max_list_length

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, Integer, RecallDataset


def trial_intrusion_error_rate(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
) -> Bool[Array, " study_positions"]:
    """Flag intrusion errors at each study position for one trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        Recall sequence for a trial. 1-indexed; 0 pads.
    presentations : Integer[Array, " study_positions"]
        Presented item IDs for the trial. 1-indexed.

    Returns
    -------
    Bool[Array, " study_positions"]
        True where the recalled item is not on the study list.

    """
    positions = jnp.arange(presentations.size)
    responses = jnp.take(recalls, positions, mode="fill", fill_value=0)
    addressed_items = presentations[jnp.clip(responses - 1, 0, presentations.size - 1)]
    valid_responses = (
        (responses > 0) & (responses <= presentations.size) & (addressed_items > 0)
    )
    return (responses != 0) & ~valid_responses & (presentations > 0)


def intrusion_error_rate(dataset: RecallDataset) -> Float[Array, " study_positions"]:
    """Return position-specific intrusion error rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean intrusion rate at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    return vmap(trial_intrusion_error_rate, in_axes=(0, 0))(
        recalls, presentations
    ).mean(axis=0)


def plot_intrusion_error_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot intrusion error rate curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Matplotlib Axes with intrusion error rate curves.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets, so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        # We'll apply the accurate_spc function to each subject, then stack
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(intrusion_error_rate),
            )
        )

        # Plot
        color = color_cycle[data_index % len(color_cycle)]
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_mean_curve(
            axis,
            xvals,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(axis, "Study Position", "Intrusion Error Rate", contrast_name)
    return axis
