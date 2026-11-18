"""Where recall stops in the sequence of responses.

After studying A, B, C, a participant who recalls C, then A, then stops has an
output length of two and a stop at output position 3. The first zero in the
recorded sequence marks that stop. A sequence that fills every recorded slot
also counts in the final bin, so that bin includes trials whose stopping time
was never observed.

``simple_termination_probability`` gives the fraction of all trials that stop
at each position. ``conditional_termination_probability`` divides the same stop
counts by the number of trials that reached each position. ``plot_termination_probability``
averages participant curves and shows the conditional probability by default.
For a single length summary per participant, ``subject_output_length_mean`` and
``subject_output_length_median`` summarize the number of responses per trial.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import jit
from matplotlib.axes import Axes

from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = [
    "conditional_termination_probability",
    "plot_termination_probability",
    "simple_termination_probability",
    "subject_output_length_mean",
    "subject_output_length_median",
]


def simple_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Termination probability by recall position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls``.

    Returns
    -------
    Float[Array, " recall_positions"]
        Probability of stopping at each output position.

    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    counts = jnp.bincount(termination_index, length=recalls.shape[1])
    return counts / recalls.shape[0]


def conditional_termination_probability(
    dataset: RecallDataset,
) -> Float[Array, " recall_positions"]:
    """Conditional termination probability by recall position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls``.

    Returns
    -------
    Float[Array, " recall_positions"]
        P(stop | reached) at each output position.

    """
    recalls = dataset["recalls"]
    zero_mask = recalls == 0
    has_zero = jnp.any(zero_mask, axis=1)
    first_zero = jnp.argmax(zero_mask, axis=1)
    last_index = recalls.shape[1] - 1
    termination_index = jnp.where(has_zero, first_zero, last_index)
    stop_counts = jnp.bincount(termination_index, length=recalls.shape[1])
    recall_positions = jnp.arange(recalls.shape[1])
    reached_mask = recall_positions <= termination_index[:, None]
    reached_counts = jnp.sum(reached_mask, axis=0)
    return stop_counts / reached_counts


def plot_termination_probability(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    mode: str = "conditional",
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot termination probability curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    mode : str
        ``"conditional"`` or ``"simple"``.
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
        Matplotlib Axes with termination curves.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    max_recall_length = max(
        int(dataset["recalls"].shape[1])
        for dataset, mask in zip(datasets, trial_masks)
        if jnp.any(mask)
    )

    if mode == "conditional":
        curve_fn = jit(conditional_termination_probability)
    elif mode == "simple":
        curve_fn = jit(simple_termination_probability)

    for index, data in enumerate(datasets):
        subject_curves = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[index],
                curve_fn,
            )
        )

        subject_curves = subject_curves[:, :max_recall_length]
        recall_positions = jnp.arange(max_recall_length, dtype=jnp.int32) + 1

        color = color_cycle[index % len(color_cycle)]
        plot_mean_curve(
            axis,
            recall_positions,
            subject_curves,
            labels[index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    ylabel = "P(Terminate | Reach)" if mode == "conditional" else "P(Terminate)"
    format_analysis_axes(axis, "Recall Position", ylabel, contrast_name)
    return axis


def _output_lengths(
    dataset: RecallDataset,
) -> Integer[Array, " trial_count"]:
    """Return output lengths per trial."""
    recalls = dataset["recalls"]
    return jnp.sum(recalls > 0, axis=1)


def _mean_output_length(
    dataset: RecallDataset,
) -> Float[Array, ""]:
    """Return the mean output length for a dataset slice."""
    return jnp.mean(_output_lengths(dataset))


def _median_output_length(
    dataset: RecallDataset,
) -> Float[Array, ""]:
    """Return the median output length for a dataset slice."""
    return jnp.median(_output_lengths(dataset))


def subject_output_length_mean(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
) -> Float[np.ndarray, " subjects"]:
    """Mean output lengths per subject.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.

    Returns
    -------
    np.ndarray
        Per-subject mean output lengths.

    """
    subject_values = apply_by_subject(dataset, trial_mask, _mean_output_length)
    return np.asarray(subject_values, dtype=float)


def subject_output_length_median(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
) -> Float[np.ndarray, " subjects"]:
    """Median output lengths per subject.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing trial data.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.

    Returns
    -------
    np.ndarray
        Per-subject median output lengths.

    """
    subject_values = apply_by_subject(dataset, trial_mask, _median_output_length)
    return np.asarray(subject_values, dtype=float)
