"""Lag-CRP with conditional transition filtering.

Extends ``crp.Tabulation`` with a per-event ``_should_tabulate`` mask
so that only a subset of transitions contribute to actual and available
lag counts, while all recalls still update availability tracking. This
supports analyses that condition on properties of the transition (e.g.,
lag direction, item category, or repetition status).

The ``Tabulation`` class shares its state layout and update logic with
``crp.Tabulation``; see that module for conventions and update steps.

"""

__all__ = [
    "set_false_at_index",
    "Tabulation",
    "tabulate_trial",
    "crp",
    "plot_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean availability vector.
    i : Int_
        1-based index to clear; ``0`` is a no-op sentinel.

    Returns
    -------
    tuple[Bool[Array, " positions"], None]
        Updated vector and ``None``.

    """

    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class Tabulation(Pytree):
    """Per-transition state for conditional Lag-CRP."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for study positions of ``recall``.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Bool[Array, " positions"]
            Updated availability vector.

        """
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """Compute unique lags from previous positions to ``recall_pos``.

        Parameters
        ----------
        recall_pos : Int_
            Study position of the current recall.

        Returns
        -------
        Bool[Array, " positions"]
            Boolean lag vector with True at each unique lag.

        """

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - prev + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        """Tabulate actual transition lags for a recall event.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated actual lag counts.

        """
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Identify recallable lag transitions from ``pos``.

        Parameters
        ----------
        pos : Int_
            Study position of the previous recall.

        Returns
        -------
        Bool[Array, " lags"]
            Boolean lag vector of available transitions.

        """
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        """Union of lags from each previous position to available items.

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated available lag counts.

        """
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when recall maps to an available position.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Bool
            Whether the recall is valid.

        """

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def tabulate(self, recall: Int_, should_tabulate: Bool_) -> "Tabulation":
        """Update state and optionally count the transition.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).
        should_tabulate : Bool_
            Whether to count this transition.

        Returns
        -------
        Tabulation
            Updated tabulation state.

        """

        def _update_state() -> "Tabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_lags=self.tabulate_actual_lags(recall),
                    avail_lags=self.tabulate_available_lags(),
                )

            def _without_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall sequence as within-list positions.
    presentation : Integer[Array, " study_events"]
        Study presentation order for the trial.
    should_tabulate : Bool[Array, " recall_events"]
        Boolean mask; True counts the transition.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(
        1, trial.size, lambda i, t: t.tabulate(trial[i], should_tabulate[i]), init
    )
    return tab.actual_lags, tab.avail_lags


def crp(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Compute conditional Lag-CRP with repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``, and
        ``_should_tabulate``.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1; NaN where denominator is zero.

    """
    should_tabulate = jnp.asarray(dataset["_should_tabulate"], dtype=bool)  # type: ignore

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        should_tabulate,
        size,
    )
    return actual.sum(0) / possible.sum(0)


def plot_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise conditional Lag-CRP with error bounds.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to plot.
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing matplotlib Axes to plot on.
    size : int, optional
        Max study positions an item can occupy.
    confidence_level : float, optional
        Confidence level for the bounds.

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
        plot_data(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
