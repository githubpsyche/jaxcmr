"""Prior-conditioned repetition cue lag-CRP.

Stratifies ordinary forward transition lags by the prior recalled
item's study-list offset from repeated-item occurrences. For each
transition, the x-axis remains the ordinary transition lag from the
just-recalled item. The access-to-repeater question is the diagonal
where transition lag equals the negative cue offset.

"""

__all__ = [
    "set_false_at_index",
    "RepCueCRPTabulation",
    "tabulate_trial",
    "repcuecrp",
    "subject_rep_cue_crp",
    "plot_rep_cue_crp",
    "plot_rep_cue_access_crp",
    "plot_first_rep_cue_access_crp",
    "plot_second_rep_cue_access_crp",
    "plot_rep_cue_crp_surface",
    "plot_rep_cue_offset_surface",
    "test_rep_cue_crp_vs_control",
    "test_rep_cue_access_vs_control",
    "test_rep_cue_access_band_vs_control",
    "test_first_second_bias",
    "test_first_second_access_bias",
    "test_first_second_access_band_bias",
    "RepCueCRPTestResult",
]

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy import stats
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to False using 1-based indexing.

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean vector of available positions.
    i : Int_
        1-based index to clear; 0 is a no-op sentinel.

    Returns
    -------
    tuple
        Updated vector and ``None`` carry value.

    """
    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class RepCueCRPTabulation(Pytree):
    """Tabulate CRPs conditional on prior cue offset from repeaters.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Presented item indices (1-indexed; 0 = pad).
    first_recall : Int_
        First recalled study position.
    min_lag : int, optional
        Minimum spacing between repeated presentations.
    size : int, optional
        Maximum study positions per item.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        size: int = 2,
    ):
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.base_matrix = jnp.zeros(
            (self.lag_range * 2 + 1, self.lag_range * 2 + 1), dtype=int
        )
        self.item_indices = jnp.arange(self.list_length, dtype=int)
        self.repetition_indices = jnp.arange(size, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        first_positions = self.item_study_positions[:, 0]
        second_positions = self.item_study_positions[:, 1]
        self.spaced_repeaters = (second_positions - first_positions) > min_lag

        self.actual_lags = jnp.zeros(
            (size, self.lag_range * 2 + 1, self.lag_range * 2 + 1), dtype=int
        )
        self.avail_lags = jnp.zeros(
            (size, self.lag_range * 2 + 1, self.lag_range * 2 + 1), dtype=int
        )

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Update the study positions available to retrieve after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def available_lags_from(
        self, prev: Int_
    ) -> Integer[Array, " transition_lags"]:
        """Return available transition lags from ``prev``."""
        return lax.cond(
            prev > 0,
            lambda: self.base_lags.at[self.all_positions - prev + self.lag_range].add(
                self.avail_recalls
            ),
            lambda: self.base_lags,
        )

    def actual_lags_from(
        self, prev: Int_, recall: Int_
    ) -> Integer[Array, " transition_lags"]:
        """Return actual transition lags from ``prev`` to ``recall``."""
        recall_study_positions = self.item_study_positions[recall - 1]

        def f(pos):
            return lax.cond(
                (pos > 0) & (prev > 0),
                lambda: self.base_lags.at[pos - prev + self.lag_range].add(1),
                lambda: self.base_lags,
            )

        return lax.map(f, recall_study_positions).sum(0).astype(bool)

    def cue_lags_from_previous(
        self, center: Int_, valid_target: Bool, prev: Int_, recall: Int_
    ) -> tuple[
        Integer[Array, " cue_offsets transition_lags"],
        Integer[Array, " cue_offsets transition_lags"],
    ]:
        """Return actual and available lag matrices for one cue center."""
        cue_offset = prev - center
        cue_index = jnp.clip(cue_offset + self.lag_range, 0, self.lag_range * 2)
        valid_cue = valid_target & (center > 0) & (prev > 0) & (cue_offset != 0)

        actual = self.actual_lags_from(prev, recall)
        possible = self.available_lags_from(prev)

        return lax.cond(
            valid_cue,
            lambda: (
                self.base_matrix.at[cue_index].add(actual),
                self.base_matrix.at[cue_index].add(possible),
            ),
            lambda: (self.base_matrix, self.base_matrix),
        )

    def tabulate_occurrence_lags(
        self, item_index: Int_, repetition_index: Int_, recall: Int_
    ) -> tuple[
        Integer[Array, " cue_offsets transition_lags"],
        Integer[Array, " cue_offsets transition_lags"],
    ]:
        """Return lag matrices for one repeated item occurrence."""
        positions = self.item_study_positions[item_index]
        is_first_occurrence = (item_index + 1) == positions[0]
        valid_target = self.spaced_repeaters[item_index] & is_first_occurrence
        center = positions[repetition_index]

        actual, possible = lax.map(
            lambda prev: self.cue_lags_from_previous(
                center, valid_target, prev, recall
            ),
            self.previous_positions,
        )
        return actual.sum(0).astype(bool), possible.sum(0).astype(bool)

    def tabulate_item_lags(
        self, item_index: Int_, recall: Int_
    ) -> tuple[
        Integer[Array, " size cue_offsets transition_lags"],
        Integer[Array, " size cue_offsets transition_lags"],
    ]:
        """Return lag matrices for one repeated item."""
        return lax.map(
            lambda repetition_index: self.tabulate_occurrence_lags(
                item_index, repetition_index, recall
            ),
            self.repetition_indices,
        )

    def tabulate_lags(
        self, recall: Int_
    ) -> tuple[
        Integer[Array, " size cue_offsets transition_lags"],
        Integer[Array, " size cue_offsets transition_lags"],
    ]:
        """Return cumulative actual and available lag counts."""
        actual, possible = lax.map(
            lambda item_index: self.tabulate_item_lags(item_index, recall),
            self.item_indices,
        )
        return (
            self.actual_lags + actual.sum(0).astype(bool),
            self.avail_lags + possible.sum(0).astype(bool),
        )

    def conditional_tabulate(self, recall: Int_) -> "RepCueCRPTabulation":
        """Tabulate transition lags and update previous/available states."""
        actual_lags, avail_lags = self.tabulate_lags(recall)
        return self.replace(
            previous_positions=self.item_study_positions[recall - 1],
            avail_recalls=self.available_recalls_after(recall),
            actual_lags=actual_lags,
            avail_lags=avail_lags,
        )

    def tabulate(self, recall: Int_) -> "RepCueCRPTabulation":
        """Update lag counts for a recall index, ignoring ``0`` sentinels."""
        return lax.cond(recall, lambda: self.conditional_tabulate(recall), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    size: int = 2,
) -> tuple[
    Float[Array, " size cue_offsets transition_lags"],
    Float[Array, " size cue_offsets transition_lags"],
]:
    """Tabulate observed and available cue-conditioned transition lags.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events for a single trial.
    presentation : Integer[Array, " study_events"]
        Study events for the trial.
    min_lag : int, optional
        Minimum spacing between item repetitions.
    size : int, optional
        Maximum positions an item can occupy.

    Returns
    -------
    tuple of Float[Array, " size cue_offsets transition_lags"]
        Actual and available lag tabulations.

    """
    init = RepCueCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def repcuecrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, " size cue_offsets transition_lags"]:
    """Prior-conditioned repetition cue lag-CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int, optional
        Minimum spacing between repeated presentations.
    size : int, optional
        Maximum study positions per item.

    Returns
    -------
    Float[Array, " size cue_offsets transition_lags"]
        CRP of shape ``(size, 2*L - 1, 2*L - 1)``.

    """
    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentations, min_lag, size
    )
    return actual.sum(0) / possible.sum(0)


class RepCueCRPWindowTabulation(Pytree):
    """Tabulate a fixed cue-offset/transition-lag window."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        max_lag: int = 5,
        max_offset: int = 5,
        size: int = 2,
    ):
        self.size = size
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.max_offset = max_offset
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(max_lag * 2 + 1, dtype=int)
        self.base_matrix = jnp.zeros(
            (max_offset * 2 + 1, max_lag * 2 + 1), dtype=int
        )
        self.item_indices = jnp.arange(self.list_length, dtype=int)
        self.repetition_indices = jnp.arange(size, dtype=int)
        self.previous_indices = jnp.arange(size, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        first_positions = self.item_study_positions[:, 0]
        second_positions = self.item_study_positions[:, 1]
        self.spaced_repeaters = (second_positions - first_positions) > min_lag

        self.actual_lags = jnp.zeros(
            (size, max_offset * 2 + 1, max_lag * 2 + 1), dtype=int
        )
        self.avail_lags = jnp.zeros(
            (size, max_offset * 2 + 1, max_lag * 2 + 1), dtype=int
        )

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Update the study positions available to retrieve after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def available_lags_from(self, prev: Int_) -> Integer[Array, " transition_lags"]:
        """Return available transition lags from ``prev`` within the window."""
        transition_lags = self.all_positions - prev
        valid_lags = (jnp.abs(transition_lags) <= self.max_lag) & (prev > 0)
        indices = jnp.clip(transition_lags + self.max_lag, 0, self.max_lag * 2)
        return self.base_lags.at[indices].add(self.avail_recalls & valid_lags)

    def actual_lags_from(
        self, prev: Int_, recall: Int_
    ) -> Integer[Array, " transition_lags"]:
        """Return actual transition lags from ``prev`` to ``recall`` within window."""
        recall_study_positions = self.item_study_positions[recall - 1]

        def f(pos):
            transition_lag = pos - prev
            valid_lag = (
                (pos > 0) & (prev > 0) & (jnp.abs(transition_lag) <= self.max_lag)
            )
            lag_index = jnp.clip(transition_lag + self.max_lag, 0, self.max_lag * 2)
            return lax.cond(
                valid_lag,
                lambda: self.base_lags.at[lag_index].add(1),
                lambda: self.base_lags,
            )

        return lax.map(f, recall_study_positions).sum(0).astype(bool)

    def cue_lags_from_previous(
        self,
        center: Int_,
        valid_target: Bool,
        prev: Int_,
        actual: Integer[Array, " transition_lags"],
        possible: Integer[Array, " transition_lags"],
    ) -> tuple[
        Integer[Array, " cue_offsets transition_lags"],
        Integer[Array, " cue_offsets transition_lags"],
    ]:
        """Return actual and available lag matrices for one cue center."""
        cue_offset = prev - center
        cue_index = jnp.clip(cue_offset + self.max_offset, 0, self.max_offset * 2)
        valid_cue = (
            valid_target
            & (center > 0)
            & (prev > 0)
            & (cue_offset != 0)
            & (jnp.abs(cue_offset) <= self.max_offset)
        )

        return lax.cond(
            valid_cue,
            lambda: (
                self.base_matrix.at[cue_index].add(actual),
                self.base_matrix.at[cue_index].add(possible),
            ),
            lambda: (self.base_matrix, self.base_matrix),
        )

    def tabulate_occurrence_lags(
        self,
        item_index: Int_,
        repetition_index: Int_,
        actual_by_prev: Integer[Array, " previous transition_lags"],
        possible_by_prev: Integer[Array, " previous transition_lags"],
    ) -> tuple[
        Integer[Array, " cue_offsets transition_lags"],
        Integer[Array, " cue_offsets transition_lags"],
    ]:
        """Return lag matrices for one repeated item occurrence."""
        positions = self.item_study_positions[item_index]
        is_first_occurrence = (item_index + 1) == positions[0]
        valid_target = self.spaced_repeaters[item_index] & is_first_occurrence
        center = positions[repetition_index]

        actual, possible = lax.map(
            lambda prev_index: self.cue_lags_from_previous(
                center,
                valid_target,
                self.previous_positions[prev_index],
                actual_by_prev[prev_index],
                possible_by_prev[prev_index],
            ),
            self.previous_indices,
        )
        return actual.sum(0).astype(bool), possible.sum(0).astype(bool)

    def tabulate_item_lags(
        self,
        item_index: Int_,
        actual_by_prev: Integer[Array, " previous transition_lags"],
        possible_by_prev: Integer[Array, " previous transition_lags"],
    ) -> tuple[
        Integer[Array, " size cue_offsets transition_lags"],
        Integer[Array, " size cue_offsets transition_lags"],
    ]:
        """Return lag matrices for one repeated item."""
        return lax.map(
            lambda repetition_index: self.tabulate_occurrence_lags(
                item_index,
                repetition_index,
                actual_by_prev,
                possible_by_prev,
            ),
            self.repetition_indices,
        )

    def tabulate_lags(
        self, recall: Int_
    ) -> tuple[
        Integer[Array, " size cue_offsets transition_lags"],
        Integer[Array, " size cue_offsets transition_lags"],
    ]:
        """Return cumulative actual and available lag counts."""
        actual_by_prev = lax.map(
            lambda prev: self.actual_lags_from(prev, recall), self.previous_positions
        )
        possible_by_prev = lax.map(self.available_lags_from, self.previous_positions)

        actual, possible = lax.map(
            lambda item_index: self.tabulate_item_lags(
                item_index, actual_by_prev, possible_by_prev
            ),
            self.item_indices,
        )
        return (
            self.actual_lags + actual.sum(0).astype(bool),
            self.avail_lags + possible.sum(0).astype(bool),
        )

    def conditional_tabulate(self, recall: Int_) -> "RepCueCRPWindowTabulation":
        """Tabulate transition lags and update previous/available states."""
        actual_lags, avail_lags = self.tabulate_lags(recall)
        return self.replace(
            previous_positions=self.item_study_positions[recall - 1],
            avail_recalls=self.available_recalls_after(recall),
            actual_lags=actual_lags,
            avail_lags=avail_lags,
        )

    def tabulate(self, recall: Int_) -> "RepCueCRPWindowTabulation":
        """Update lag counts for a recall index, ignoring ``0`` sentinels."""
        return lax.cond(recall, lambda: self.conditional_tabulate(recall), lambda: self)


def _tabulate_trial_window(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: int = 5,
    size: int = 2,
) -> tuple[
    Float[Array, " size cue_offsets transition_lags"],
    Float[Array, " size cue_offsets transition_lags"],
]:
    """Tabulate observed and available lags inside a fixed window."""
    init = RepCueCRPWindowTabulation(
        presentation, trial[0], min_lag, max_lag, max_offset, size
    )
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def _repcuecrp_window(
    dataset: RecallDataset,
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: int = 5,
    size: int = 2,
) -> Float[Array, " size cue_offsets transition_lags"]:
    """Prior-conditioned repetition cue lag-CRP inside a fixed window."""
    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    actual, possible = vmap(
        _tabulate_trial_window, in_axes=(0, 0, None, None, None, None)
    )(trials, presentations, min_lag, max_lag, max_offset, size)
    return actual.sum(0) / possible.sum(0)


def subject_rep_cue_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    size: int = 2,
) -> np.ndarray:
    """Compute subject-level prior-conditioned repetition cue CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    min_lag : int, optional
        Minimum spacing between item repetitions.
    max_lag : int, optional
        Maximum transition lag to include in output.
    max_offset : int, optional
        Maximum cue offset to include in output.
    size : int, optional
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size, 2*max_offset+1, 2*max_lag+1)``.

    """
    if max_offset is None:
        max_offset = max_lag

    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(
            _repcuecrp_window,
            static_argnames=("min_lag", "max_lag", "max_offset", "size"),
        ),
        min_lag=min_lag,
        max_lag=max_lag,
        max_offset=max_offset,
        size=size,
    )
    return np.stack(subject_values)


def _subject_rep_cue_crp_full(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    size: int = 2,
) -> np.ndarray:
    """Compute subject-level cue CRP by slicing the full tabulation."""
    if max_offset is None:
        max_offset = max_lag

    lag_range = dataset["pres_itemnos"].shape[1] - 1
    offset_slice = slice(lag_range - max_offset, lag_range + max_offset + 1)
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(repcuecrp, static_argnames=("min_lag", "size")),
        min_lag=min_lag,
        size=size,
    )
    return np.stack([s[:, offset_slice, lag_slice] for s in subject_values])



def _access_diagonal(
    crp: np.ndarray,
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract access-to-repeater diagonal values over cue offsets.

    Offset 0 is retained as an all-NaN gap so access plots break at the
    repeated item, matching ordinary lag-CRP plotting conventions.

    """
    cue_offsets = np.arange(-max_offset, max_offset + 1)
    lag_labels = np.arange(-max_lag, max_lag + 1)
    access = np.full((*crp.shape[:2], cue_offsets.size), np.nan)

    for offset_idx, offset in enumerate(cue_offsets):
        if offset == 0:
            continue
        lag_match = np.where(lag_labels == -offset)[0]
        if lag_match.size:
            source_idx = offset + max_offset
            access[:, :, offset_idx] = crp[:, :, source_idx, lag_match[0]]

    return cue_offsets, access


def _select_cue_offset(
    crp: np.ndarray,
    cue_offset: int,
    max_offset: int = 5,
) -> np.ndarray:
    """Select one cue-offset CRP curve from subject-level arrays."""
    offset_index = cue_offset + max_offset
    return crp[:, :, offset_index, :]


def _masked_cmap(cmap: str = "viridis"):
    """Return colormap with NaN cells shown in neutral light gray."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f2f2f2")
    return cmap_obj


def _overlay_access_cells(
    axis: Axes,
    max_offset: int = 5,
    max_lag: int = 5,
    edgecolor: str = "white",
) -> Axes:
    """Outline cells where the next recall lands on the repeater."""
    for cue_offset in range(-max_offset, max_offset + 1):
        transition_lag = -cue_offset
        if cue_offset == 0 or abs(transition_lag) > max_lag:
            continue
        axis.add_patch(
            Rectangle(
                (transition_lag - 0.5, cue_offset - 0.5),
                1,
                1,
                fill=False,
                edgecolor=edgecolor,
                linewidth=1.75,
            )
        )
    return axis


def _previous_next_surface(
    crp: np.ndarray,
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform cue-offset/transition-lag CRP to previous/next offsets.

    The input's last two dimensions are ``cue_offset`` and
    ``transition_lag``. The returned array's last two dimensions are
    ``next_offset`` and ``previous_offset`` so it can be plotted with
    previous offset on the x-axis and next offset on the y-axis.

    """
    offsets = np.arange(-max_offset, max_offset + 1)
    transition_lags = np.arange(-max_lag, max_lag + 1)
    transformed = np.full((*crp.shape[:-2], offsets.size, offsets.size), np.nan)

    for previous_index, previous_offset in enumerate(offsets):
        for lag_index, transition_lag in enumerate(transition_lags):
            next_offset = previous_offset + transition_lag
            if -max_offset <= next_offset <= max_offset:
                next_index = next_offset + max_offset
                transformed[..., next_index, previous_index] = crp[
                    ..., previous_index, lag_index
                ]

    return offsets, transformed


def plot_rep_cue_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    cue_offset: int = 1,
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot transition-lag CRP for one prior cue offset.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Dataset(s) containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    cue_offset : int, optional
        Prior-item offset from the repeated occurrence.
    max_lag : int, optional
        Maximum transition lag to display.
    max_offset : int, optional
        Maximum cue offset in the subject-level slice.
    min_lag : int, optional
        Minimum spacing between repeated occurrences.
    size : int, optional
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

    Returns
    -------
    Axes
        Axes with the cue-conditioned CRP plot.

    """
    if max_offset is None:
        max_offset = max_lag

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
        subject_values = subject_rep_cue_crp(
            data,
            trial_masks[data_index],
            min_lag=min_lag,
            max_lag=max_lag,
            max_offset=max_offset,
            size=size,
        )
        cue_values = _select_cue_offset(subject_values, cue_offset, max_offset)

        for rep_idx, repetition_index in enumerate(repetition_indices):
            repetition_subject_values = cue_values[:, repetition_index, :]
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

    set_plot_labels(axis, "Transition Lag", "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot the access-to-repeater diagonal across cue offsets."""
    if max_offset is None:
        max_offset = max_lag

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    labels_list = list(labels) if labels is not None else []
    if isinstance(repetition_index, int):
        repetition_indices = [repetition_index]
    else:
        repetition_indices = list(range(size))

    for data_index, data in enumerate(datasets):
        subject_values = subject_rep_cue_crp(
            data,
            trial_masks[data_index],
            min_lag=min_lag,
            max_lag=max_lag,
            max_offset=max_offset,
            size=size,
        )
        cue_offsets, access_values = _access_diagonal(
            subject_values, max_offset=max_offset, max_lag=max_lag
        )

        for rep_idx, repetition_index in enumerate(repetition_indices):
            repetition_subject_values = access_values[:, repetition_index, :]
            label = (
                labels_list[data_index]
                if len(repetition_indices) == 1 and data_index < len(labels_list)
                else labels_list[repetition_index]
                if len(datasets) == 1 and repetition_index < len(labels_list)
                else str(repetition_index + 1)
            )
            color_idx = data_index * len(repetition_indices) + rep_idx
            color = color_cycle[color_idx % len(color_cycle)]
            plot_data(
                axis,
                cue_offsets,
                repetition_subject_values,
                label,
                color,
                confidence_level=confidence_level,
            )

    set_plot_labels(axis, "Cue Offset", "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_first_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Convenience function to plot first-presentation cue access CRP."""
    return plot_rep_cue_access_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        max_offset=max_offset,
        min_lag=min_lag,
        size=size,
        repetition_index=0,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_second_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Convenience function to plot second-presentation cue access CRP."""
    return plot_rep_cue_access_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        max_offset=max_offset,
        min_lag=min_lag,
        size=size,
        repetition_index=1,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_rep_cue_crp_surface(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: int = 0,
    axis: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mark_access: bool = True,
    colorbar: bool = True,
    cmap: str = "viridis",
) -> Axes:
    """Plot the full cue-offset x transition-lag CRP surface."""
    if max_offset is None:
        max_offset = max_lag
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        dataset, trial_mask, None, axis
    )
    del datasets, trial_masks, color_cycle

    subject_values = subject_rep_cue_crp(
        dataset,
        trial_mask,
        min_lag=min_lag,
        max_lag=max_lag,
        max_offset=max_offset,
        size=size,
    )
    surface = np.nanmean(subject_values[:, repetition_index], axis=0)
    image = axis.imshow(
        np.ma.masked_invalid(surface),
        origin="lower",
        aspect="auto",
        cmap=_masked_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        extent=[
            -max_lag - 0.5,
            max_lag + 0.5,
            -max_offset - 0.5,
            max_offset + 0.5,
        ],
    )
    if colorbar:
        axis.figure.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(-max_lag, max_lag + 1))
    axis.set_yticks(np.arange(-max_offset, max_offset + 1))
    axis.set_xlabel("Next Lag from Previous", fontsize=16)
    axis.set_ylabel("Previous Offset from R", fontsize=16)
    axis.tick_params(labelsize=14)
    if mark_access:
        _overlay_access_cells(axis, max_offset=max_offset, max_lag=max_lag)
    return axis


def plot_rep_cue_offset_surface(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: Optional[int] = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: int = 0,
    axis: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mark_repeater: bool = True,
    mark_forward: bool = True,
    colorbar: bool = True,
    cmap: str = "viridis",
) -> Axes:
    """Plot previous-offset x next-offset CRP surface."""
    if max_offset is None:
        max_offset = max_lag
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        dataset, trial_mask, None, axis
    )
    del datasets, trial_masks, color_cycle

    subject_values = subject_rep_cue_crp(
        dataset,
        trial_mask,
        min_lag=min_lag,
        max_lag=max_lag,
        max_offset=max_offset,
        size=size,
    )
    offsets, offset_values = _previous_next_surface(
        subject_values, max_offset=max_offset, max_lag=max_lag
    )
    surface = np.nanmean(offset_values[:, repetition_index], axis=0)
    image = axis.imshow(
        np.ma.masked_invalid(surface),
        origin="lower",
        aspect="auto",
        cmap=_masked_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        extent=[
            -max_offset - 0.5,
            max_offset + 0.5,
            -max_offset - 0.5,
            max_offset + 0.5,
        ],
    )
    if colorbar:
        axis.figure.colorbar(image, ax=axis)
    axis.set_xticks(offsets)
    axis.set_yticks(offsets)
    axis.set_xlabel("Previous Offset from R", fontsize=16)
    axis.set_ylabel("Next Offset from R", fontsize=16)
    axis.tick_params(labelsize=14)

    if mark_repeater:
        axis.axhline(
            0,
            color="white",
            linewidth=1.75,
        )
    if mark_forward:
        x_values = offsets
        y_values = offsets + 1
        valid = (y_values >= -max_offset) & (y_values <= max_offset)
        axis.plot(
            x_values[valid],
            y_values[valid],
            linestyle="--",
            color="black",
            linewidth=1.25,
            alpha=0.8,
        )
    return axis


@dataclass
class RepCueCRPTestResult:
    """Results from a prior-conditioned repetition cue CRP test."""

    lags: np.ndarray
    t_stats: np.ndarray
    t_pvals: np.ndarray
    w_stats: np.ndarray
    w_pvals: np.ndarray
    mean_diffs: np.ndarray
    label_name: str = "Lag"

    def __str__(self) -> str:
        label_width = max(
            5,
            len(self.label_name),
            *(len(str(label)) for label in self.lags),
        )
        lines = [
            f"{self.label_name:>{label_width}} | {'t-stat':>8} {'t p-val':>10} | "
            f"{'W-stat':>8} {'W p-val':>10} | {'Mean Diff':>10}",
            f"{'-'*label_width}-+-{'-'*20}-+-{'-'*20}-+-{'-'*11}",
        ]
        for i, lag in enumerate(self.lags):
            lines.append(
                f"{str(lag):>{label_width}} | "
                f"{self.t_stats[i]:>8.3f} {self.t_pvals[i]:>10.4f} | "
                f"{self.w_stats[i]:>8.1f} {self.w_pvals[i]:>10.4f} | "
                f"{self.mean_diffs[i]:>10.4f}"
            )
        return "\n".join(lines)


def _paired_tests(
    obs: np.ndarray,
    ctrl: np.ndarray,
    labels: np.ndarray,
    label_name: str = "Lag",
) -> RepCueCRPTestResult:
    """Return paired tests for aligned observed/control columns."""
    n_lags = len(labels)
    t_stats = np.zeros(n_lags)
    t_pvals = np.zeros(n_lags)
    w_stats = np.zeros(n_lags)
    w_pvals = np.zeros(n_lags)
    mean_diffs = np.zeros(n_lags)

    for lag_idx in range(n_lags):
        obs_col = obs[:, lag_idx]
        ctrl_col = ctrl[:, lag_idx]
        diff = obs_col - ctrl_col
        valid = ~(np.isnan(obs_col) | np.isnan(ctrl_col))

        if valid.sum() > 1:
            t_stat, t_pval = stats.ttest_rel(obs_col, ctrl_col, nan_policy="omit")
        else:
            t_stat, t_pval = np.nan, np.nan
        t_stats[lag_idx] = t_stat
        t_pvals[lag_idx] = t_pval

        if valid.sum() > 10:
            w_stat, w_pval = stats.wilcoxon(diff[valid], alternative="two-sided")
        else:
            w_stat, w_pval = np.nan, np.nan
        w_stats[lag_idx] = w_stat
        w_pvals[lag_idx] = w_pval
        mean_diffs[lag_idx] = np.nanmean(diff) if valid.any() else np.nan

    return RepCueCRPTestResult(
        lags=labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
        label_name=label_name,
    )


def _format_offset_band(cue_offsets: Sequence[int]) -> str:
    """Return a compact label for an offset band."""
    return ",".join(f"{offset:+d}" for offset in cue_offsets)


def _access_band(
    crp: np.ndarray,
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Average access-diagonal values across selected nonzero offsets."""
    cue_offsets = tuple(cue_offsets)
    if any(offset == 0 for offset in cue_offsets):
        raise ValueError("cue_offsets for an access band must be nonzero.")

    all_offsets, access = _access_diagonal(
        crp, max_offset=max_offset, max_lag=max_lag
    )
    offset_indices = []
    for cue_offset in cue_offsets:
        matches = np.where(all_offsets == cue_offset)[0]
        if not matches.size:
            raise ValueError(f"cue_offset {cue_offset} is outside the plotted range.")
        offset_indices.append(matches[0])

    selected = access[:, :, offset_indices]
    valid = np.isfinite(selected)
    counts = valid.sum(axis=2)
    summed = np.nansum(selected, axis=2)
    band = np.full(counts.shape, np.nan)
    band[counts > 0] = summed[counts > 0] / counts[counts > 0]
    return np.asarray(cue_offsets), band


def test_rep_cue_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    cue_offset: int,
    max_offset: int = 5,
    max_lag: int = 5,
) -> dict[str, RepCueCRPTestResult]:
    """Test observed vs control cue-conditioned CRP per index.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP, shape
        ``(n_subjects, size, 2*max_offset+1, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level control CRP, same shape.
    cue_offset : int
        Cue offset whose transition-lag CRP should be tested.
    max_offset : int, optional
        Maximum cue offset in the CRP arrays.
    max_lag : int, optional
        Maximum transition lag used for labeling.

    Returns
    -------
    dict[str, RepCueCRPTestResult]
        Results keyed by presentation label.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    observed = _select_cue_offset(observed_crp, cue_offset, max_offset)
    control = _select_cue_offset(control_crp, cue_offset, max_offset)
    size = observed.shape[1]
    results = {}

    for rep_idx in range(size):
        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = _paired_tests(
            observed[:, rep_idx, :], control[:, rep_idx, :], lag_labels
        )

    return results


def test_rep_cue_access_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_offset: int = 5,
    max_lag: int = 5,
) -> dict[str, RepCueCRPTestResult]:
    """Test observed vs control access diagonal per index."""
    cue_offsets, observed = _access_diagonal(
        observed_crp, max_offset=max_offset, max_lag=max_lag
    )
    _, control = _access_diagonal(control_crp, max_offset=max_offset, max_lag=max_lag)
    size = observed.shape[1]
    results = {}

    for rep_idx in range(size):
        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = _paired_tests(
            observed[:, rep_idx, :],
            control[:, rep_idx, :],
            cue_offsets,
            label_name="Offset",
        )

    return results


def test_rep_cue_access_band_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> dict[str, RepCueCRPTestResult]:
    """Test observed vs control access averaged across selected offsets."""
    selected_offsets, observed = _access_band(
        observed_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    _, control = _access_band(
        control_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    labels = np.array([_format_offset_band(selected_offsets)])
    size = observed.shape[1]
    results = {}

    for rep_idx in range(size):
        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = _paired_tests(
            observed[:, rep_idx, None],
            control[:, rep_idx, None],
            labels,
            label_name="Offsets",
        )

    return results


def test_first_second_bias(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    cue_offset: int,
    max_offset: int = 5,
    max_lag: int = 5,
) -> RepCueCRPTestResult:
    """Test whether first-presentation cue CRP bias differs from control."""
    lag_labels = np.arange(-max_lag, max_lag + 1)
    observed = _select_cue_offset(observed_crp, cue_offset, max_offset)
    control = _select_cue_offset(control_crp, cue_offset, max_offset)

    observed_diff = observed[:, 0, :] - observed[:, 1, :]
    control_diff = control[:, 0, :] - control[:, 1, :]
    return _paired_tests(observed_diff, control_diff, lag_labels)


def test_first_second_access_bias(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_offset: int = 5,
    max_lag: int = 5,
) -> RepCueCRPTestResult:
    """Test whether first-presentation access bias differs from control."""
    cue_offsets, observed = _access_diagonal(
        observed_crp, max_offset=max_offset, max_lag=max_lag
    )
    _, control = _access_diagonal(control_crp, max_offset=max_offset, max_lag=max_lag)

    observed_diff = observed[:, 0, :] - observed[:, 1, :]
    control_diff = control[:, 0, :] - control[:, 1, :]
    return _paired_tests(
        observed_diff,
        control_diff,
        cue_offsets,
        label_name="Offset",
    )


def test_first_second_access_band_bias(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> RepCueCRPTestResult:
    """Test first-second access bias averaged across selected offsets."""
    selected_offsets, observed = _access_band(
        observed_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    _, control = _access_band(
        control_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )

    observed_diff = observed[:, 0] - observed[:, 1]
    control_diff = control[:, 0] - control[:, 1]
    return _paired_tests(
        observed_diff[:, None],
        control_diff[:, None],
        np.array([_format_offset_band(selected_offsets)]),
        label_name="Offsets",
    )
