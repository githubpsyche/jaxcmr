"""Cue-centered Lag-CRP.

Computes lag-CRP where lags are measured from a retrieval cue to the
recalled item, rather than from the previously recalled item. Designed
for paradigms where participants receive an external cue at each
recall event (e.g., video-recall tasks with clip cues).

Notes
-----
- Cue identity is supplied per recall event via ``cue_clips``;
  each cue is mapped to its study positions the same way recalls
  are, allowing repeated cue items.
- A ``_should_tabulate`` mask controls which events contribute to
  lag counts (e.g., to exclude uncued or practice events).

"""

from __future__ import annotations

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset

__all__ = [
    "CueCenteredTabulation",
    "tabulate_trial",
    "cue_centered_crp",
    "plot_cue_centered_crp",
    "set_false_at_index",
]


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


class CueCenteredTabulation(Pytree):
    """Per-transition state for cue-centered Lag-CRP."""

    def __init__(self, presentation: Integer[Array, " study_events"], size: int = 3):
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.all_items = jnp.arange(1, self.list_length + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda item: item_to_study_positions(item, presentation, size),
            self.all_items,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)

    def item_positions(self, item: Int_) -> Integer[Array, " size"]:
        """Return study positions for an item identifier.

        Parameters
        ----------
        item : Int_
            Item identifier; ``0`` returns all zeros.

        Returns
        -------
        Integer[Array, " size"]
            Study positions occupied by the item.

        """
        zeros = jnp.zeros_like(self.item_study_positions[0])
        return lax.cond(
            (item > 0) & (item <= self.list_length),
            lambda: self.item_study_positions[item - 1],
            lambda: zeros,
        )

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for study positions of ``recall``.

        Parameters
        ----------
        recall : Int_
            Recalled item identifier.

        Returns
        -------
        Bool[Array, " positions"]
            Updated availability vector.

        """
        study_positions = self.item_positions(recall)
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when recall maps to an available position.

        Parameters
        ----------
        recall : Int_
            Recalled item identifier.

        Returns
        -------
        Bool
            Whether the recall is valid.

        """

        def _for_nonzero():
            recall_study_positions = self.item_positions(recall)
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def lags_from_cue(
        self,
        cue_pos: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> Bool[Array, " lags"]:
        """Return unique lags from a cue position to recall positions.

        Parameters
        ----------
        cue_pos : Int_
            Study position of the cue item.
        recall_positions : Integer[Array, " size"]
            Study positions of the recalled item.

        Returns
        -------
        Bool[Array, " lags"]
            Boolean lag vector with True at each unique lag.

        """

        def f(recall_pos):
            return lax.cond(
                (cue_pos * recall_pos) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - cue_pos + self.lag_range].add(1),
            )

        return lax.map(f, recall_positions).sum(0).astype(bool)

    def tabulate_actual_lags(
        self,
        cue_positions: Integer[Array, " size"],
        recall: Int_,
    ) -> Integer[Array, " lags"]:
        """Tabulate cue-centered lags for the current recall.

        Parameters
        ----------
        cue_positions : Integer[Array, " size"]
            Study positions of the cue item.
        recall : Int_
            Recalled item identifier.

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated actual lag counts.

        """
        recall_positions = self.item_positions(recall)
        new_lags = (
            lax.map(
                lambda pos: self.lags_from_cue(pos, recall_positions), cue_positions
            )
            .sum(0)
            .astype(bool)
        )
        return self.actual_lags + new_lags

    def available_lags_from_cue(self, cue_pos: Int_) -> Bool[Array, " lags"]:
        """Identify available lags from a cue position.

        Parameters
        ----------
        cue_pos : Int_
            Study position of the cue item.

        Returns
        -------
        Bool[Array, " lags"]
            Boolean lag vector of available transitions.

        """
        return lax.cond(
            cue_pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[
                self.all_positions - cue_pos + self.lag_range
            ].add(self.avail_recalls),
        )

    def tabulate_available_lags(
        self, cue_positions: Integer[Array, " size"]
    ) -> Integer[Array, " lags"]:
        """Tabulate available lags from cue positions.

        Parameters
        ----------
        cue_positions : Integer[Array, " size"]
            Study positions of the cue item.

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated available lag counts.

        """
        new_lags = (
            lax.map(self.available_lags_from_cue, cue_positions).sum(0).astype(bool)
        )
        return self.avail_lags + new_lags

    def tabulate(
        self, recall: Int_, cue: Int_, should_tabulate: Bool_
    ) -> "CueCenteredTabulation":
        """Update state and optionally count the transition.

        Parameters
        ----------
        recall : Int_
            Recalled item identifier.
        cue : Int_
            Cue item identifier for this recall event.
        should_tabulate : Bool_
            Whether to count this transition.

        Returns
        -------
        CueCenteredTabulation
            Updated tabulation state.

        """

        def _update_state() -> "CueCenteredTabulation":
            new_avail_recalls = self.available_recalls_after(recall)
            cue_positions = self.item_positions(cue)
            has_cue = jnp.any(cue_positions != 0)

            def _with_counts() -> "CueCenteredTabulation":
                return self.replace(
                    avail_recalls=new_avail_recalls,
                    actual_lags=self.tabulate_actual_lags(cue_positions, recall),
                    avail_lags=self.tabulate_available_lags(cue_positions),
                )

            def _without_counts() -> "CueCenteredTabulation":
                return self.replace(
                    avail_recalls=new_avail_recalls,
                )

            should_count = should_tabulate & has_cue
            return lax.cond(should_count, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    recalls: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    cues: Integer[Array, " recall_events"],
    should_tabulate: Bool[Array, " recall_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate cue-centered actual and available lags for a trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_events"]
        Recall sequence as item identifiers.
    presentation : Integer[Array, " study_events"]
        Study presentation order for the trial.
    cues : Integer[Array, " recall_events"]
        Cue item identifiers aligned to recall events.
    should_tabulate : Bool[Array, " recall_events"]
        Boolean mask; True counts the transition.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """
    init = CueCenteredTabulation(presentation, size)
    tab = lax.fori_loop(
        0,
        recalls.size,
        lambda i, t: t.tabulate(recalls[i], cues[i], should_tabulate[i]),
        init,
    )
    return tab.actual_lags, tab.avail_lags


def cue_centered_crp(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Compute cue-centered Lag-CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``,
        ``cue_clips``, and ``_should_tabulate``.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1; NaN where denominator is zero.

    """
    should_tabulate = jnp.asarray(dataset["_should_tabulate"], dtype=bool)  # type: ignore
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        dataset["cue_clips"],
        should_tabulate,
        size,
    )
    return actual.sum(0) / possible.sum(0)


def plot_cue_centered_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    should_tabulate: (
        Sequence[Bool[Array, " trial_count recall_events"]]
        | Bool[Array, " trial_count recall_events"]
    ),
    max_lag: int = 5,
    exclude_zero_lag: bool = False,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot cue-centered Lag-CRP curves.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks selecting trials in datasets.
    should_tabulate : Sequence | array
        Boolean masks aligned to recall events.
    max_lag : int, optional
        Maximum lag to plot.
    exclude_zero_lag : bool, optional
        Whether to omit the zero-lag bin from plotting.
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
        Matplotlib Axes with cue-centered Lag-CRP plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    if not isinstance(should_tabulate, Sequence):
        should_tabulate = [jnp.array(should_tabulate)]

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        list_length = int(data["pres_itemnos"].shape[1])
        lag_range = list_length - 1
        slice_start = lag_range - max_lag
        slice_end = lag_range + max_lag + 1
        data_with_mask = {
            **data,
            "_should_tabulate": should_tabulate[data_index],
        }
        subject_values = apply_by_subject(
            data_with_mask,  # type: ignore
            trial_masks[data_index],
            jit(cue_centered_crp, static_argnames=("size",)),
            size=size,
        )
        subject_values = jnp.vstack(subject_values)
        subject_values = subject_values[:, slice_start:slice_end]
        if exclude_zero_lag:
            center_index = max_lag
            subject_values = subject_values.at[:, center_index].set(jnp.nan)

        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Cue-Centered Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
