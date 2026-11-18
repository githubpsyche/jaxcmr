"""Conditional response probability by lag between successive recalls.

Suppose a participant studies A, B, C, D and recalls B followed by C. That
transition has lag +1 because C was studied one position after B. From B, the
participant could also have recalled A (lag −1) or D (lag +2). The trial
therefore contributes one actual transition at +1 and one opportunity at
each of −1, +1 and +2.

``tabulate_trial`` counts actual and available lags. ``crp`` sums both across
trials and divides at each lag. A lag with no opportunities has an undefined
probability. ``plot_crp`` calculates probabilities per participant before
averaging them.

For a list with repeated items, one recall refers to all presentations of
its item. A transition can then contribute at several lags. Each distinct
lag counts at most once per transition in each tally. For lists of distinct
items, ``simple_crp`` needs only the position-coded recalls and list length.

Use complete study lists of the same length in a call. The numerical
calculations can be JIT-compiled with a fixed list width and occurrence
capacity (``size``).
"""

__all__ = [
    "SimpleTabulation",
    "Tabulation",
    "crp",
    "plot_crp",
    "simple_crp",
    "simple_tabulate_trial",
    "tabulate_trial",
]

from collections.abc import Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import lag_counts, remove_positions


class SimpleTabulation(Pytree):
    """CRP tabulator for lists without repeated items."""

    def __init__(self, list_length: int, first_recall: Int_) -> None:
        self.lag_range = list_length - 1
        self.all_items = jnp.arange(1, list_length + 1, dtype=int)
        self.actual_transitions = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_transitions = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_items = jnp.ones(list_length, dtype=bool)
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall

    def _update(self, current_item: Int_) -> "SimpleTabulation":
        "Tabulate actual and possible serial lags of current from previous item."
        actual_lag = current_item - self.previous_item + self.lag_range
        all_lags = self.all_items - self.previous_item + self.lag_range

        return self.replace(
            previous_item=current_item,
            avail_items=self.avail_items.at[current_item - 1].set(False),
            avail_transitions=self.avail_transitions.at[all_lags].add(self.avail_items),
            actual_transitions=self.actual_transitions.at[actual_lag].add(1),
        )

    def update(self, choice: Int_) -> "SimpleTabulation":
        "Tabulate a transition if the choice is non-zero (i.e., a valid item)."
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def simple_tabulate_trial(
    trial: Integer[Array, " recall_events"], list_length: int
) -> SimpleTabulation:
    """Tabulate lag transitions for a single no-repeat trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in 1..L with 0 pads.
    list_length : int
        Study-list length.

    Returns
    -------
    SimpleTabulation
        Accumulated actual and available transitions.

    """
    return lax.scan(
        lambda tabulation, recall: (tabulation.update(recall), None),
        SimpleTabulation(list_length, trial[0]),
        trial[1:],
    )[0]


def simple_crp(
    trials: Integer[Array, "trials recall_events"], list_length: int
) -> Float[Array, " lags"]:
    """Compute Lag-CRP across multiple no-repeat trials.

    Parameters
    ----------
    trials : Integer[Array, "trials recall_events"]
        Serial positions in 1..L with 0 pads.
    list_length : int
        Study-list length.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1, indexed by lag offset.

    """
    tabulated_trials = lax.map(lambda t: simple_tabulate_trial(t, list_length), trials)
    total_actual_transitions = jnp.sum(tabulated_trials.actual_transitions, axis=0)
    total_possible_transitions = jnp.sum(tabulated_trials.avail_transitions, axis=0)
    return total_actual_transitions / total_possible_transitions


class Tabulation(Pytree):
    """CRP tabulator supporting repeated items."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ) -> None:
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for all study positions of a recall."""
        study_positions = self.item_study_positions[recall - 1]
        return remove_positions(self.avail_recalls, study_positions)

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """Unique lags from previous positions to a recall position."""

        def lags_from_previous_position(
            previous_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                (recall_pos * previous_position) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[
                    recall_pos - previous_position + self.lag_range
                ].add(1),
            )

        return (
            lax.map(lags_from_previous_position, self.previous_positions)
            .sum(0)
            .astype(bool)
        )

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulates the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identifies recallable lag transitions from the specified study position."
        return lag_counts(
            pos, self.all_positions, self.avail_recalls, self.all_positions.size - 1
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Union of lags from each previous position to all currently available positions."
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self, recall: Int_) -> Bool[Array, ""]:
        "Only consider transitions from items with study positions that have not been recalled yet."

        def _for_nonzero() -> Bool[Array, ""]:
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def tabulate(self, recall: Int_) -> "Tabulation":
        "Tabulates actual and possible serial lags of current from previous item."
        return lax.cond(
            self.should_tabulate(recall),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
            ),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in 1..L with 0 pads.
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    size : int
        Max study positions per item.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def crp(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Compute Lag-CRP with support for repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1; NaN where denominator is
        zero.

    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"], dataset["pres_itemnos"], size
    )
    return actual.sum(0) / possible.sum(0)


def plot_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    size: int = 3,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot subject-wise Lag-CRP with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
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
        Matplotlib Axes with the Lag-CRP plot.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = (jnp.max(data["listLength"]) - 1).item()
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(crp, static_argnames=("size")),
            size=size,
        )
        subject_values = jnp.vstack(subject_values)
        subject_values = subject_values[
            :, lag_range - max_lag : lag_range + max_lag + 1
        ]
        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
