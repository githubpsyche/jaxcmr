# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
# ---

"""Lag of the first qualifying transition from a repeated item during serial recall.

For the list A, B, X, C, D, E, F, G, X, H, suppose the participant starts
A, B, X, H. The run up to X follows study order. The transition from X
to H is scored even though H breaks that order. It contributes lag +7
from the first X and +1 from the second. Each trial supplies at most one
such transition. The first recall initializes the run, and order checks begin
with the second response. To require a correct first recall, select those
trials before calculating the curve.

``repcrp`` pools the actual and possible lags from each presentation across
trials. In this serial measure the possible lags are based on the full
study layout. The free-recall measure in ``analyses.repcrp`` removes recalled
items from the available alternatives.
"""

# %% auto 0
__all__ = [
    "RepCRPTabulation",
    "plot_difference_rep_crp",
    "plot_first_rep_crp",
    "plot_rep_crp",
    "plot_second_rep_crp",
    "repcrp",
    "subject_serial_rep_crp",
    "tabulate_trial",
]


from collections.abc import Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import lag_counts


class RepCRPTabulation(Pytree):
    """Tabulate serial repetition lag transitions.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Presented item indices (1-indexed; 0 = pad).
    first_recall : Int_
        Study position of the first recalled item.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int,
        size: int,
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

        self.previous_positions = lax.cond(
            first_recall > 0,
            lambda: self.item_study_positions[first_recall - 1],
            lambda: jnp.zeros((self.size,), dtype=int),
        )
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        # self.avail_recalls = self.available_recalls_after(first_recall)

        # #! Add flag to indicate if we are done tabulating this trial
        self.has_tabulated = jnp.bool_(False)
        self.has_errored = jnp.bool_(False)

    # for updating avail_recalls: study positions still available for retrieval
    # def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
    #     "Update the study positions available to retrieve after a transition."
    #     study_positions = self.item_study_positions[recall - 1]
    #     return remove_positions(self.avail_recalls, study_positions)

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, pos: Int_) -> Bool[Array, " size positions"]:
        "Identify the lag(s) from the study position(s) of the previous item."

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
        "Tabulate the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]

        # modified to sum over current_recall axis while still separating over prev_recall
        new_lags = lax.map(self.lags_from_previous, recall_study_positions).astype(bool)
        return self.actual_lags + jnp.any(new_lags, axis=0)  # new_lags.sum(0)

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identify recallable lag transitions from the specified study position."
        return lag_counts(
            pos, self.all_positions, self.avail_recalls, self.all_positions.size - 1
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Tabulate available transitions after a transition."

        # modified to not sum over prev_recall axis before summing
        new_lags = lax.map(self.available_lags_from, self.previous_positions)
        return self.avail_lags + new_lags.astype(bool)

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool[Array, ""]:
        "Only consider transitions from item with at least two spaced-out study positions"
        spacing = (self.previous_positions.size > 1) & (
            (self.previous_positions[-1] - self.previous_positions[-2]) > self.min_lag
        )
        return spacing & (~self.has_tabulated) & (~self.has_errored)

    def conditional_tabulate(
        self, recall: Int_, recall_idx: Int_
    ) -> "RepCRPTabulation":
        "Only tabulate actual and possible lags if the additional condition is met."

        # #! here we will call is_out_of_order using current recall to configure has_errored
        recall_study_positions = self.item_study_positions[recall - 1]
        has_errored = jnp.logical_or(
            ~jnp.any(recall_study_positions == recall_idx + 1), self.has_errored
        )

        return lax.cond(
            self.should_tabulate(),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                # avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
                # #! Set flag to indicate that we have tabulated this trial
                has_tabulated=True,
                # #! Set flag to indicate if current recall is out of order
                has_errored=has_errored,
            ),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                # avail_recalls=self.available_recalls_after(recall),
                # #! Set flag to indicate if current recall is out of order
                has_errored=has_errored,
            ),
        )

    def tabulate(self, recall: Int_, recall_idx: Int_) -> "RepCRPTabulation":
        "Tabulate actual and possible serial lags of current from previous item."
        return lax.cond(
            recall,
            lambda: self.conditional_tabulate(recall, recall_idx),
            lambda: self,
        )


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
    min_lag : int, optional
        Minimum spacing between item repetitions.
    size : int, optional
        Maximum presentations per item.

    Returns
    -------
    tuple of Float[Array, " lags"]
        Actual and available lag tabulations.

    """
    init = RepCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i], i), init)
    return tab.actual_lags, tab.avail_lags


def repcrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, " lags"]:
    """Serial repetition lag-CRP per presentation index.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int, optional
        Minimum separation between repeated presentations.
    size : int, optional
        Maximum presentations per item.

    Returns
    -------
    Float[Array, " lags"]
        CRP of shape ``(size, 2*L-1)`` per repetition
        index.

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
    max_lag: int = 3,
    min_lag: int = 2,
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
    """Plot serial repetition lag-CRP with CIs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to display.
    min_lag : int, optional
        Minimum spacing between repeated occurrences.
    size : int, optional
        Maximum presentations per item.
    repetition_index : int or None, optional
        Plot only this repetition index.
    color_cycle : list[str] or None, optional
        Colors for each curve.
    labels : Sequence[str] or None, optional
        Legend labels for repetition-index lines.
    contrast_name : str or None, optional
        Legend title.
    axis : Axes or None, optional
        Existing Axes to plot on.
    confidence_level : float, optional
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Axes with the serial repetition CRP plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    labels_list = list(labels) if labels is not None else []

    if isinstance(repetition_index, int):
        repetition_indices = [repetition_index]
    else:
        repetition_indices = list(range(size))

    # lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)
    lower_bound = 1
    lag_interval = jnp.arange(lower_bound, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        trial_mask = trial_masks[data_index].reshape(-1)
        if not bool(jnp.any(trial_mask)):
            raise ValueError("No trials selected by trial_mask.")
        lag_range = int(jnp.max(data["listLength"][trial_mask])) - 1
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
            )[:, lag_range + lower_bound : lag_range + max_lag + 1]

            label = (
                labels_list[selected_repetition_index]
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
    max_lag: int = 3,
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
    """Plot first-minus-second serial repetition CRP.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to display.
    min_lag : int, optional
        Minimum spacing between repeated occurrences.
    size : int, optional
        Maximum presentations per item.
    color_cycle : list[str] or None, optional
        Colors for each curve.
    labels : Sequence[str] or None, optional
        Legend labels for each dataset.
    contrast_name : str or None, optional
        Legend title.
    axis : Axes or None, optional
        Existing Axes to plot on.
    confidence_level : float, optional
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

    lower_bound = 1
    lag_interval = jnp.arange(lower_bound, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        trial_mask = trial_masks[data_index].reshape(-1)
        if not bool(jnp.any(trial_mask)):
            raise ValueError("No trials selected by trial_mask.")
        lag_range = int(jnp.max(data["listLength"][trial_mask])) - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repcrp, static_argnames=("size",)),
            min_lag,
            size,
        )
        first = jnp.vstack([each[0] for each in subject_values])[
            :, lag_range + lower_bound : lag_range + max_lag + 1
        ]
        second = jnp.vstack([each[1] for each in subject_values])[
            :, lag_range + lower_bound : lag_range + max_lag + 1
        ]
        diff_subject_values = first - second

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_interval,
            diff_subject_values,
            labels[data_index],
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
    max_lag: int = 3,
    min_lag: int = 2,
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
    max_lag: int = 3,
    min_lag: int = 2,
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
        repetition_index=1,  # first repetition
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def subject_serial_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 2,
    max_lag: int = 3,
    size: int = 2,
) -> Float[np.ndarray, "subjects occurrences lags"]:
    """Compute subject-level serial repetition CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials to include.
    min_lag : int, optional
        Minimum spacing between item repetitions.
    max_lag : int, optional
        Maximum lag to include in output.
    size : int, optional
        Maximum presentations per item.

    """
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range + 1, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset, trial_mask, jit(repcrp, static_argnames=("size",)), min_lag, size
    )
    return np.stack([s[:, lag_slice] for s in subject_values])
