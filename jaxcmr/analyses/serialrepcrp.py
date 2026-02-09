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

"""Serial repetition CRP conditioned on correct run-up.

Computes lag-CRP restricted to transitions that follow a correct
serial run-up, measuring how repetition affects forward transitions
when recall is already proceeding in order.

"""

# %% auto 0
__all__ = [
    "set_false_at_index",
    "RepCRPTabulation",
    "tabulate_trial",
    "repcrp",
    "plot_rep_crp",
    "plot_difference_rep_crp",
    "plot_first_rep_crp",
    "plot_second_rep_crp",
    "subject_serial_rep_crp",
    "test_serial_rep_crp_vs_control",
]

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy import stats
from simple_pytree import Pytree

from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

def set_false_at_index(vec: Bool[Array, " positions"], i: Int_):
    return lax.cond(i, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None))


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
    ):
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
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
    #     return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, pos: Int_) -> Bool[Array, " size positions"]:
        "Identify the lag(s) from the study position(s) of the previous item."

        def f(prev):
            return lax.cond(
                (pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[pos - prev + self.lag_range].add(1),
            )

        # modified to not sum over prev_recall axis before returning
        return lax.map(f, self.previous_positions).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulate the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]

        # modified to sum over current_recall axis while still separating over prev_recall
        new_lags = lax.map(self.lags_from_previous, recall_study_positions).astype(bool)
        return self.actual_lags + jnp.any(new_lags, axis=0)#new_lags.sum(0)

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identify recallable lag transitions from the specified study position."
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Tabulate available transitions after a transition."

        # modified to not sum over prev_recall axis before summing
        new_lags = lax.map(self.available_lags_from, self.previous_positions)
        return self.avail_lags + new_lags.astype(bool)

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool:
        "Only consider transitions from item with at least two spaced-out study positions"
        spacing = (
            (self.previous_positions.size > 1) &
            ((self.previous_positions[-1] - self.previous_positions[-2]) > self.min_lag)
        )
        return spacing & (~self.has_tabulated) & (~self.has_errored)

    def conditional_tabulate(self, recall: Int_, recall_idx: Int_) -> "RepCRPTabulation":
        "Only tabulate actual and possible lags if the additional condition is met."

        # #! here we will call is_out_of_order using current recall to configure has_errored
        recall_study_positions = self.item_study_positions[recall - 1]
        has_errored = jnp.logical_or(~jnp.any(recall_study_positions == recall_idx + 1), self.has_errored)

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
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
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

        for rep_idx, repetition_index in enumerate(repetition_indices):
            repetition_subject_values = jnp.vstack(
                [each[repetition_index] for each in subject_values]
            )[:, lag_range + lower_bound : lag_range + max_lag + 1]

            label = (
                labels_list[repetition_index]
                if len(datasets) == 1 and repetition_index < len(labels_list)
                else str(repetition_index + 1)
            )
            color_idx = data_index * len(repetition_indices) + rep_idx
            color = color_cycle[color_idx % len(color_cycle)]
            plot_data(
                axis,
                lag_interval,
                repetition_subject_values,
                label,
                color,
                confidence_level=confidence_level,
            )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_difference_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
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
        plot_data(
            axis,
            lag_interval,
            diff_subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Lag", "Diff. Conditional Resp. Prob.", contrast_name)
    return axis


def plot_first_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 2,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    "Convenience function to plot first repetition CRP."
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
    )


def plot_second_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 2,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    "Convenience function to plot first repetition CRP."
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
    )


def subject_serial_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 2,
    max_lag: int = 3,
    size: int = 2,
) -> np.ndarray:
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


@dataclass
class SerialRepCRPTestResult:
    """Results from a serial repetition CRP statistical test."""

    lags: np.ndarray
    t_stats: np.ndarray
    t_pvals: np.ndarray
    w_stats: np.ndarray
    w_pvals: np.ndarray
    mean_diffs: np.ndarray

    def __str__(self) -> str:
        lines = [
            f"{'Lag':>5} | {'t-stat':>8} {'t p-val':>10} | "
            f"{'W-stat':>8} {'W p-val':>10} | {'Mean Diff':>10}",
            f"{'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*11}",
        ]
        for i, lag in enumerate(self.lags):
            lines.append(
                f"{lag:>5} | {self.t_stats[i]:>8.3f} {self.t_pvals[i]:>10.4f} | "
                f"{self.w_stats[i]:>8.1f} {self.w_pvals[i]:>10.4f} | "
                f"{self.mean_diffs[i]:>10.4f}"
            )
        return "\n".join(lines)


def test_serial_rep_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 3,
) -> dict[str, SerialRepCRPTestResult]:
    """Test observed vs control serial rep CRP per index.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, size, max_lag)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, size, max_lag)``.
    max_lag : int, optional
        Maximum lag used for labeling.

    """
    lag_labels = np.arange(1, max_lag + 1)
    n_lags = len(lag_labels)
    size = observed_crp.shape[1]
    results: dict[str, SerialRepCRPTestResult] = {}

    for rep_idx in range(size):
        obs = observed_crp[:, rep_idx, :]
        ctrl = control_crp[:, rep_idx, :]

        t_stats = np.zeros(n_lags)
        t_pvals = np.zeros(n_lags)
        w_stats = np.zeros(n_lags)
        w_pvals = np.zeros(n_lags)
        mean_diffs = np.zeros(n_lags)

        for lag_idx in range(n_lags):
            obs_col = obs[:, lag_idx]
            ctrl_col = ctrl[:, lag_idx]
            diff = obs_col - ctrl_col

            t_stat, t_pval = stats.ttest_rel(obs_col, ctrl_col, nan_policy="omit")
            t_stats[lag_idx] = t_stat
            t_pvals[lag_idx] = t_pval

            valid = ~(np.isnan(obs_col) | np.isnan(ctrl_col))
            if valid.sum() > 10:
                w_stat, w_pval = stats.wilcoxon(diff[valid], alternative="two-sided")
            else:
                w_stat, w_pval = np.nan, np.nan
            w_stats[lag_idx] = w_stat
            w_pvals[lag_idx] = w_pval
            mean_diffs[lag_idx] = np.nanmean(diff)

        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = SerialRepCRPTestResult(
            lags=lag_labels,
            t_stats=t_stats,
            t_pvals=t_pvals,
            w_stats=w_stats,
            w_pvals=w_pvals,
            mean_diffs=mean_diffs,
        )

    return results
