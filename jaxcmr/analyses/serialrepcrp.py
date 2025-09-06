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

# %% auto 0
__all__ = ['set_false_at_index', 'RepCRPTabulation', 'tabulate_trial', 'repcrp', 'plot_rep_crp', 'plot_difference_rep_crp',
           'plot_first_rep_crp', 'plot_second_rep_crp']

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

def set_false_at_index(vec: Bool[Array, " positions"], i: Int_):
    return lax.cond(i, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None))


class RepCRPTabulation(Pytree):
    "A tabulation of transitions between items during recall of a study list."

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

        self.previous_positions = self.item_study_positions[first_recall - 1]
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
    min_lag=4,
    size: int = 2,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    init = RepCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i], i), init)
    return tab.actual_lags, tab.avail_lags

def repcrp(
    trials: Integer[Array, "trials recall_events"],
    presentation: Integer[Array, "trials study_events"],
    list_length: int,
    min_lag=4,
    size: int = 2,
) -> Float[Array, " lags"]:
    """Returns lag-CRP centered around each study position of repeated items across trials.

    Args:
        trials: Recall events for each trial.
        presentation: Study events for each trial.
        list_length: Number of study events in each trial; unused.
        min_lag: Minimum amount of study positions between two presentations of an item.
        size: Maximum number of study positions an item can be presented
    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentation, min_lag, size
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
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes object with plotted prob of repetition lag-crp for datasets and trial masks.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        max_lag: Maximum number of study positions an item can be presented at.
        min_lag: Minimum amount of study positions between two presentations of an item.
        size: Maximum number of study positions an item can be presented at.
        color_cycle: List of colors for plotting each dataset.
        distances: Unused, included for compatibility with other plotting functions.
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

    # lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)
    lower_bound = 1
    lag_interval = jnp.arange(lower_bound, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repcrp, static_argnames=("size",)),
            min_lag,
            size,
        )

        for repetition_index in repetition_indices:
            rep_mat = jnp.vstack(
                [each[repetition_index] for each in subject_values]
            )#[:, lag_range - max_lag : lag_range + max_lag + 1]
            repetition_subject_values = rep_mat[:, lag_range + lower_bound: lag_range + max_lag + 1]

            if len(datasets) == 1:
                label = labels[repetition_index] or str(repetition_index + 1)
            else:
                label = labels[data_index] or str(repetition_index + 1)
            color = color_cycle.pop(0)
            plot_data(
                axis,
                lag_interval,
                repetition_subject_values,
                label,
                color,
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
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes object with plotted prob of repetition lag-crp for datasets and trial masks.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        max_lag: Maximum number of study positions an item can be presented at.
        min_lag: Minimum amount of study positions between two presentations of an item.
        size: Maximum number of study positions an item can be presented at.
        color_cycle: List of colors for plotting each dataset.
        distances: Unused, included for compatibility with other plotting functions.
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

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
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

        color = color_cycle.pop(0)
        plot_data(
            axis,
            lag_interval,
            repetition_subject_values,
            str(repetition_index + 1),
            color,
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
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
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
        distances=distances,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
    )


def plot_second_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 2,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
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
        distances=distances,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
    )
