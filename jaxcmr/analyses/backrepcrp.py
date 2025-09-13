"""Lag-CRP for repeated items.

Extends the lag conditional response probability (CRP) analysis to repeated
study items. Tabulation restricts transitions to recalls of repeated items
and separates counts by repetition index.
"""

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

__all__ = [
    "set_false_at_index",
    "RepCRPTabulation",
    "tabulate_trial",
    "repcrp",
    "plot_back_rep_crp",
    "plot_difference_rep_crp",
    "plot_first_rep_crp",
    "plot_second_rep_crp",
]


def set_false_at_index(vec: Bool[Array, " positions"], i: Int_):
    """Set an index to ``False`` when positive.

    Args:
      vec: Vector to update. Shape [positions].
      i: One-based index where ``0`` is a sentinel.

    Returns:
      (updated, None): Updated vector and placeholder flag.
    """
    return lax.cond(i, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None))


class RepCRPTabulation(Pytree):
    """Tabulates lag transitions between items during recall."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        size: int = 3,
    ):
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size), self.all_positions
        )

        self.actual_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)
        self.avail_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Returns flags for study positions available after a recall.

        Args:
          recall: Recalled item index (1-based).
        """
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def lags_from_previous(self, pos: Int_) -> Bool[Array, " size positions"]:
        """Returns lag indicators from previous item positions.

        Args:
          pos: Current item's study position.
        """

        def f(prev):
            return lax.cond(
                (pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[prev - pos + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        """Returns updated counts for lags made to the recalled item.

        Args:
          recall: Item index of the current recall.
        """
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = lax.map(self.lags_from_previous, recall_study_positions).astype(bool)
        return self.actual_lags + new_lags.sum(0)

    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Returns recallable lag transitions from a study position.

        Args:
          pos: Study position to transition from.
        """
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[pos - self.all_positions + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        """Returns updated counts for recallable lags from previous item."""
        new_lags = lax.map(self.available_lags_from, self.previous_positions)
        return self.avail_lags + new_lags.astype(bool)

    def should_tabulate(self, recall) -> Bool:
        """Returns True when recall item has spaced study positions.

        Args:
          recall: Item index of the current recall.
        """
        recall_study_positions = self.item_study_positions[recall - 1]
        return (
            len(recall_study_positions) > 1
            and recall_study_positions[-1] - recall_study_positions[-2] > self.min_lag
        )

    def conditional_tabulate(self, recall: Int_) -> "RepCRPTabulation":
        """Returns updated tabulation when recall meets spacing criteria.

        Args:
          recall: Item index of the current recall.
        """
        return lax.cond(
            self.should_tabulate(recall),
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
        """Returns tabulation updated with the recalled item.

        Args:
          recall: Item index of the current recall.
        """
        return lax.cond(recall, lambda: self.conditional_tabulate(recall), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag=4,
    size: int = 2,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Returns counts of actual and available lags for a trial.

    Args:
      trial: Recall events.
      presentation: Study events.
      min_lag: Minimum spacing between presentations.
      size: Maximum number of study positions per item.
    """
    init = RepCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def repcrp(
    trials: Integer[Array, "trials recall_events"],
    presentation: Integer[Array, "trials study_events"],
    list_length: int,
    min_lag=4,
    size: int = 2,
) -> Float[Array, " lags"]:
    """Returns lag-CRP centered on each study position of repeated items.

    Args:
      trials: Recall events for each trial.
      presentation: Study events for each trial.
      list_length: Number of study events in each trial; unused.
      min_lag: Minimum spacing between presentations.
      size: Maximum number of study positions per item.
    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentation, min_lag, size
    )
    return actual.sum(0) / possible.sum(0)


def plot_back_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes with repetition lag-CRP.

    Args:
      datasets: Trial data.
      trial_masks: Trial filters for each dataset.
      max_lag: Maximum study positions.
      min_lag: Minimum spacing between presentations.
      size: Maximum number of study positions per item.
      color_cycle: Colors for plotting each dataset.
      distances: Unused; retained for API compatibility.
      labels: Names for each dataset.
      contrast_name: Legend label for contrasting datasets.
      axis: Existing Matplotlib axis, optional.
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
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repcrp, static_argnames=("size",)),
            min_lag,
            size,
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


def plot_difference_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes with difference between first and second repetition lag-CRPs.

    Args:
      datasets: Trial data.
      trial_masks: Trial filters for each dataset.
      max_lag: Maximum study positions.
      min_lag: Minimum spacing between presentations.
      size: Maximum number of study positions per item.
      color_cycle: Colors for plotting each dataset.
      distances: Unused; retained for API compatibility.
      labels: Names for each dataset.
      contrast_name: Legend label for contrasting datasets.
      axis: Existing Matplotlib axis, optional.
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
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes with first repetition lag-CRP."""
    return plot_back_rep_crp(
        datasets,
        trial_masks,
        max_lag,
        min_lag,
        size,
        repetition_index=0,
        color_cycle=color_cycle,
        distances=distances,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
    )


def plot_second_rep_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    distances: Optional[Float[Array, "word_count word_count"]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Returns Axes with second repetition lag-CRP."""
    return plot_back_rep_crp(
        datasets,
        trial_masks,
        max_lag,
        min_lag,
        size,
        repetition_index=1,
        color_cycle=color_cycle,
        distances=distances,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
    )

