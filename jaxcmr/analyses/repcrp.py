"""Recall transitions out of an item studied more than once.

Consider the study list A, B, X, C, D, E, F, G, X, H. If X is recalled
and H comes next, H is seven positions after the first X and one after the
second. The same response contributes to lag +7 on the first-presentation
curve and lag +1 on the second.

``tabulate_trial`` counts actual and available lags separately from each
presentation of a qualifying repeated item. ``repcrp`` pools these counts
across trials before dividing. Comparing the curves shows where subsequent
recalls fall relative to each presentation. The observed response identifies
the item X and leaves its retrieved occurrence unknown.
"""

__all__ = [
    "RepCRPTabulation",
    "plot_difference_rep_crp",
    "plot_first_rep_crp",
    "plot_rep_crp",
    "plot_second_rep_crp",
    "repcrp",
    "subject_rep_crp",
    "tabulate_trial",
    "test_first_second_bias",
]

from collections.abc import Sequence

import numpy as np
import pandas as pd
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import lag_counts, remove_positions
from .statistics import compare_paired


class RepCRPTabulation(Pytree):
    """Tabulate lag transitions for repeated items during recall.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Item numbers at each study position (1-indexed).
    first_recall : Int_
        First recalled item number (1-indexed).
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        size: int = 3,
    ) -> None:
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)
        self.avail_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Return study positions available for retrieval after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return remove_positions(self.avail_recalls, study_positions)

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, pos: Int_) -> Bool[Array, " size positions"]:
        "Identify lags from study positions of the previous item."

        def lags_from_previous_position(
            previous_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                (pos * previous_position) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[pos - previous_position + self.lag_range].add(
                    1
                ),
            )

        # modified to not sum over prev_recall axis before returning
        return lax.map(lags_from_previous_position, self.previous_positions).astype(
            bool
        )

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulate observed lag transitions for a recall."
        recall_study_positions = self.item_study_positions[recall - 1]

        # modified to sum over current_recall axis while still separating over prev_recall
        new_lags = lax.map(self.lags_from_previous, recall_study_positions).astype(bool)
        per_lag = jnp.any(new_lags, axis=0)
        return self.actual_lags + per_lag  # new_lags.sum(0)

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Return recallable lag transitions from ``pos``."
        return lag_counts(
            pos, self.all_positions, self.avail_recalls, self.all_positions.size - 1
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Tabulate potential transitions after a recall."

        # modified to not sum over prev_recall axis before summing
        new_lags = lax.map(self.available_lags_from, self.previous_positions)
        return self.avail_lags + new_lags.astype(bool)

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool[Array, ""]:
        "Return ``True`` when the prior item was studied twice with spacing > ``min_lag``."
        return (
            len(self.previous_positions) > 1
            and self.previous_positions[-1] - self.previous_positions[-2] > self.min_lag
        )

    def conditional_tabulate(self, recall: Int_) -> "RepCRPTabulation":
        "Tabulate lags only when ``should_tabulate`` is ``True``."
        return lax.cond(
            self.should_tabulate(),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
            ),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
            ),
        )

    def tabulate(self, recall: Int_) -> "RepCRPTabulation":
        "Update lag counts for a recall index, ignoring ``0`` sentinels."
        return lax.cond(recall, lambda: self.conditional_tabulate(recall), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    size: int = 2,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate observed and available lags for a trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events for a single trial.
    presentation : Integer[Array, " study_events"]
        Study events for the trial.
    min_lag : int
        Minimum spacing between item repetitions.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag tabulations.

    """
    init = RepCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def repcrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, " lags"]:
    """Lag-CRP centered on each presentation of repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum presentations per item.

    Returns
    -------
    Float[Array, " lags"]
        CRP of shape ``(size, 2*L - 1)`` per repetition index.

    """
    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentations, min_lag, size
    )
    return actual.sum(0) / possible.sum(0)


def plot_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: int | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot repetition Lag-CRP with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum presentations per item.
    repetition_index : int, optional
        Plot only this repetition index.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels for repetition-index lines.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
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
        Axes with the repetition Lag-CRP plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    labels_list = list(labels) if labels is not None else []
    if isinstance(repetition_index, int):
        repetition_indices = [repetition_index]
    else:
        repetition_indices = list(range(size))

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = data["pres_itemnos"].shape[1] - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repcrp, static_argnames=("size",)),
            min_lag,
            size,
        )

        for rep_idx, selected_repetition_index in enumerate(repetition_indices):
            repetition_subject_values = jnp.vstack(
                [each[selected_repetition_index] for each in subject_values]
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

            label = (
                labels_list[data_index]
                if len(repetition_indices) == 1 and data_index < len(labels_list)
                else labels_list[selected_repetition_index]
                if len(datasets) == 1 and selected_repetition_index < len(labels_list)
                else str(selected_repetition_index + 1)
            )
            color_idx = data_index * len(repetition_indices) + rep_idx
            color = color_cycle[color_idx % len(color_cycle)]
            plot_mean_curve(
                axis,
                lag_interval,
                repetition_subject_values,
                label,
                color,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )

    format_analysis_axes(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_difference_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot difference between first and second presentation CRPs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum presentations per item.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
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
        Axes with the difference CRP plot.
    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = data["pres_itemnos"].shape[1] - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repcrp, static_argnames=("size",)),
            min_lag,
            size,
        )

        repetition_index = 0
        repetition_subject_values = jnp.vstack(
            [each[repetition_index] for each in subject_values]
        )[:, lag_range - max_lag : lag_range + max_lag + 1]

        repetition_index = 1
        repetition_subject_values -= jnp.vstack(
            [each[repetition_index] for each in subject_values]
        )[:, lag_range - max_lag : lag_range + max_lag + 1]

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_interval,
            repetition_subject_values,
            str(repetition_index + 1),
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(axis, "Lag", "Diff. Conditional Resp. Prob.", contrast_name)
    return axis


def plot_first_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Convenience function to plot first repetition CRP.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    return plot_rep_crp(
        datasets,
        trial_masks,
        max_lag,
        min_lag,
        size,
        repetition_index=0,  # first repetition
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_second_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Convenience function to plot second repetition CRP.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
    return plot_rep_crp(
        datasets,
        trial_masks,
        max_lag,
        min_lag,
        size,
        repetition_index=1,  # first repetition
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def subject_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> Float[np.ndarray, "subjects occurrences lags"]:
    """Compute subject-level repetition CRP values.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    min_lag : int
        Minimum spacing between item repetitions.
    max_lag : int
        Maximum lag to include in output.
    size : int
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size, 2*max_lag+1)``.

    """
    lag_range = dataset["pres_itemnos"].shape[1] - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset, trial_mask, jit(repcrp, static_argnames=("size",)), min_lag, size
    )
    return np.stack([s[:, lag_slice] for s in subject_values])


def test_first_second_bias(
    observed_crp: Float[np.ndarray, "subjects occurrences lags"],
    control_crp: Float[np.ndarray, "subjects occurrences lags"],
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare first-minus-second occurrence scores between observed and control data.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence and transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence and transition lag. Participant rows must align across inputs.
    max_lag : int, optional
        Inclusive transition-lag window; labels run from -max_lag to +max_lag.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. Rows are labeled by transition lag.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted.
    """
    observed = observed_crp[:, 0, :] - observed_crp[:, 1, :]
    control = control_crp[:, 0, :] - control_crp[:, 1, :]
    return compare_paired(observed, control, labels=np.arange(-max_lag, max_lag + 1))
