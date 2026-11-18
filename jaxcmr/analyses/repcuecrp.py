"""Transition lags grouped by proximity of the previous recall to a repeated item.

In the list A, B, X, C, D, E, F, G, X, H, B lies one position before the
first X. Transitions out of B therefore contribute to the row for cue
offset −1 on the first-presentation surface. The columns give the lag of
the next recall measured from B. For example, recalling C from B contributes
at lag +2. Each cell divides actual transitions by opportunities at that
offset and lag.

``repcuecrp`` returns a surface per presentation.
``rep_cue_access`` selects the cells where the transition lands on the
presentation itself. In the row for offset −1, this is lag +1. Selecting
these cells across offsets gives the access curve. ``rep_cue_access_band``
summarizes that curve over a chosen range of offsets.
"""

from matplotlib.colors import Colormap

__all__ = [
    "RepCueCRPWindowTabulation",
    "plot_first_rep_cue_access_crp",
    "plot_rep_cue_access_crp",
    "plot_rep_cue_crp",
    "plot_rep_cue_crp_surface",
    "plot_rep_cue_offset_surface",
    "plot_second_rep_cue_access_crp",
    "rep_cue_access",
    "rep_cue_access_band",
    "repcuecrp",
    "subject_rep_cue_crp",
    "tabulate_trial",
    "test_first_second_access_band_bias",
    "test_first_second_access_bias",
    "test_first_second_bias",
    "test_rep_cue_access_band_vs_control",
    "test_rep_cue_access_vs_control",
]

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from jaxcmr._vendor.simple_pytree import Pytree

from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import lag_counts, remove_positions
from .statistics import compare_paired


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    size: int = 2,
    *,
    max_lag: int | None = None,
    max_offset: int | None = None,
) -> tuple[
    Integer[Array, " occurrences cue_offsets lags"],
    Integer[Array, " occurrences cue_offsets lags"],
]:
    """Count actual and available cue-conditioned transitions in one trial.

    Parameters
    ----------
    trial : integer array of shape (recall_events,)
        Position-coded recalls, with zero for unused responses.
    presentation : integer array of shape (study_events,)
        Within-list item identities, including repeated presentations.
    min_lag : int, optional
        Minimum separation used to select spaced repeated items.
    size : int, optional
        Occurrence capacity, default two.
    max_lag, max_offset : int, optional
        Inclusive transition-lag and cue-offset windows. Omit for the full list
        range. Window arguments and occurrence capacity must be static under JIT.

    Returns
    -------
    tuple of integer arrays
        Actual and available counts, with axes occurrence, cue offset and lag.
        Both windows run from negative to positive values including zero.
    """
    lag_limit = presentation.size - 1 if max_lag is None else max_lag
    offset_limit = presentation.size - 1 if max_offset is None else max_offset
    initial = RepCueCRPWindowTabulation(
        presentation, trial[0], min_lag, lag_limit, offset_limit, size
    )
    final = lax.fori_loop(
        1, trial.size, lambda index, state: state.tabulate(trial[index]), initial
    )
    return final.actual_lags, final.avail_lags


def repcuecrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
    *,
    max_lag: int | None = None,
    max_offset: int | None = None,
) -> Float[Array, " occurrences cue_offsets lags"]:
    """Pool cue-conditioned transition counts across a dataset.

    Parameters
    ----------
    dataset : RecallDataset
        Lists and position-coded recalls, normally from one participant.
    min_lag : int, optional
        Minimum repeated-item separation.
    size : int, optional
        Occurrence capacity, default two.
    max_lag, max_offset : int, optional
        Inclusive windows, or the full list range when omitted. These arguments
        and ``size`` must be static under JIT.

    Returns
    -------
    array of shape (occurrences, cue_offsets, lags)
        Actual divided by available counts. Zero opportunities remain NaN.
        Use ``subject_rep_cue_crp`` for separate participant summaries.
    """
    actual, possible = vmap(
        lambda trial, presentation: tabulate_trial(
            trial, presentation, min_lag, size, max_lag=max_lag, max_offset=max_offset
        )
    )(dataset["recalls"], dataset["pres_itemnos"])
    return actual.sum(axis=0) / possible.sum(axis=0)


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
    ) -> None:
        self.size = size
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.max_offset = max_offset
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(max_lag * 2 + 1, dtype=int)
        self.base_matrix = jnp.zeros((max_offset * 2 + 1, max_lag * 2 + 1), dtype=int)
        self.item_indices = jnp.arange(self.list_length, dtype=int)
        self.repetition_indices = jnp.arange(size, dtype=int)
        self.previous_indices = jnp.arange(size, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
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
        return remove_positions(self.avail_recalls, study_positions)

    def available_lags_from(self, prev: Int_) -> Integer[Array, " transition_lags"]:
        """Return available transition lags from ``prev`` within the window."""
        return lag_counts(
            prev, self.all_positions, self.avail_recalls, self.base_lags.size // 2
        )

    def actual_lags_from(
        self, prev: Int_, recall: Int_
    ) -> Integer[Array, " transition_lags"]:
        """Return actual transition lags from ``prev`` to ``recall`` within window."""
        recall_study_positions = self.item_study_positions[recall - 1]

        def lags_to_position(
            study_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            transition_lag = study_position - prev
            valid_lag = (
                (study_position > 0)
                & (prev > 0)
                & (jnp.abs(transition_lag) <= self.max_lag)
            )
            lag_index = jnp.clip(transition_lag + self.max_lag, 0, self.max_lag * 2)
            return lax.cond(
                valid_lag,
                lambda: self.base_lags.at[lag_index].add(1),
                lambda: self.base_lags,
            )

        return lax.map(lags_to_position, recall_study_positions).sum(0).astype(bool)

    def cue_lags_from_previous(
        self,
        center: Int_,
        valid_target: Bool[Array, ""],
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


def _study_positions_by_position(
    presentation: Integer[np.ndarray, " study_events"],
    size: int = 2,
) -> Integer[np.ndarray, " study_positions_plus_one occurrences"]:
    """Return study positions for each position's item."""
    list_length = presentation.size
    positions_by_study_position = np.zeros((list_length + 1, size), dtype=int)

    for study_position in range(1, list_length + 1):
        item = presentation[study_position - 1]
        if item == 0:
            continue
        positions = np.flatnonzero(presentation == item)[:size] + 1
        positions_by_study_position[study_position, : positions.size] = positions

    return positions_by_study_position


def _prepare_repcue_presentation(
    presentation: Integer[np.ndarray, " study_events"],
    min_lag: int = 4,
    size: int = 2,
) -> tuple[
    Integer[np.ndarray, " study_positions_plus_one occurrences"],
    Bool[np.ndarray, " occurrences study_positions_plus_one"],
    Integer[np.ndarray, " study_positions"],
]:
    """Precompute repeated-item centers for one presentation row."""
    list_length = presentation.size
    positions_by_study_position = _study_positions_by_position(presentation, size)
    valid_centers = np.zeros((size, list_length + 1), dtype=bool)

    for study_position in range(1, list_length + 1):
        positions = positions_by_study_position[study_position]
        if positions.size < 2:
            continue
        valid_target = (positions[1] - positions[0]) > min_lag
        is_first_occurrence = study_position == positions[0]
        if not (valid_target and is_first_occurrence):
            continue
        for repetition_index, center in enumerate(positions):
            if center > 0:
                valid_centers[repetition_index, center] = True

    all_positions = np.arange(1, list_length + 1)
    return positions_by_study_position, valid_centers, all_positions


def _tabulate_subject_rep_cue_crp_window(
    recalls: Integer[np.ndarray, " trials recall_events"],
    presentations: Integer[np.ndarray, " trials study_events"],
    presentation_cache: dict[
        tuple[tuple[int, ...], str, bytes],
        tuple[
            Integer[np.ndarray, " study_positions_plus_one occurrences"],
            Bool[np.ndarray, " occurrences study_positions_plus_one"],
            Integer[np.ndarray, " study_positions"],
        ],
    ],
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: int = 5,
    size: int = 2,
) -> Float[np.ndarray, " occurrences cue_offsets lags"]:
    """Compute one subject's cue CRP with a fixed offset/lag window."""
    offset_labels = np.arange(-max_offset, max_offset + 1)
    actual_lags = np.zeros((size, offset_labels.size, max_lag * 2 + 1), dtype=int)
    avail_lags = np.zeros_like(actual_lags)

    for trial, presentation in zip(recalls, presentations):
        key = (presentation.shape, presentation.dtype.str, presentation.tobytes())
        if key not in presentation_cache:
            presentation_cache[key] = _prepare_repcue_presentation(
                presentation, min_lag, size
            )
        positions_by_study_position, valid_centers, all_positions = presentation_cache[
            key
        ]
        list_length = presentation.size

        first_recall = trial[0]
        if first_recall <= 0 or first_recall > list_length:
            continue

        previous_positions = positions_by_study_position[first_recall]
        avail_recalls = np.ones(list_length + 1, dtype=bool)
        avail_recalls[0] = False
        avail_recalls[previous_positions[previous_positions > 0]] = False

        for recall in trial[1:]:
            if recall <= 0 or recall > list_length:
                continue

            recall_positions = positions_by_study_position[recall]
            actual_transition = np.zeros_like(actual_lags, dtype=bool)
            avail_transition = np.zeros_like(avail_lags, dtype=bool)

            for previous_position in previous_positions:
                if previous_position <= 0:
                    continue

                transition_lags = all_positions - previous_position
                valid_lags = np.abs(transition_lags) <= max_lag
                possible_positions = avail_recalls[1:] & valid_lags
                possible_counts = np.zeros(max_lag * 2 + 1, dtype=bool)
                possible_counts[transition_lags[possible_positions] + max_lag] = True

                actual_counts = np.zeros(max_lag * 2 + 1, dtype=bool)
                for recall_position in recall_positions:
                    transition_lag = recall_position - previous_position
                    if recall_position > 0 and abs(transition_lag) <= max_lag:
                        actual_counts[transition_lag + max_lag] = True

                for offset_index, cue_offset in enumerate(offset_labels):
                    if cue_offset == 0:
                        continue
                    center = previous_position - cue_offset
                    if center <= 0 or center > list_length:
                        continue
                    repetition_indices = np.flatnonzero(valid_centers[:, center])
                    for repetition_index in repetition_indices:
                        actual_transition[repetition_index, offset_index] |= (
                            actual_counts
                        )
                        avail_transition[repetition_index, offset_index] |= (
                            possible_counts
                        )

            actual_lags += actual_transition
            avail_lags += avail_transition
            previous_positions = recall_positions
            avail_recalls[recall_positions[recall_positions > 0]] = False

    with np.errstate(divide="ignore", invalid="ignore"):
        return actual_lags / avail_lags


def subject_rep_cue_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    max_offset: int | None = None,
    size: int = 2,
) -> Float[np.ndarray, " subjects occurrences cue_offsets lags"]:
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

    subject_ids = np.asarray(dataset["subject"]).reshape(-1)
    numpy_trial_mask = np.asarray(trial_mask, dtype=bool).reshape(-1)
    recalls = np.asarray(dataset["recalls"])
    presentations = np.asarray(dataset["pres_itemnos"])
    presentation_cache: dict[
        tuple[tuple[int, ...], str, bytes],
        tuple[
            Integer[np.ndarray, " study_positions_plus_one occurrences"],
            Bool[np.ndarray, " occurrences study_positions_plus_one"],
            Integer[np.ndarray, " study_positions"],
        ],
    ] = {}
    subject_values = []

    for subject in np.unique(subject_ids):
        subject_mask = (subject_ids == subject) & numpy_trial_mask
        if subject_mask.sum() == 0:
            continue
        subject_values.append(
            _tabulate_subject_rep_cue_crp_window(
                recalls[subject_mask],
                presentations[subject_mask],
                presentation_cache,
                min_lag=min_lag,
                max_lag=max_lag,
                max_offset=max_offset,
                size=size,
            )
        )

    return np.stack(subject_values)


def rep_cue_access(
    crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[
    Integer[np.ndarray, " cue_offsets"],
    Float[np.ndarray, " subjects occurrences cue_offsets"],
]:
    """Select transitions to a repeated item from the cue-conditioned CRP.

    Parameters
    ----------
    crp : array of shape (participants, occurrences, cue_offsets, lags)
        Participant cue-conditioned probabilities. Offset and lag axes run from
        their negative limit through zero to their positive limit.
    max_offset, max_lag : int, optional
        Inclusive windows represented by the supplied array, default five each.

    Returns
    -------
    tuple of arrays
        Cue offsets and scores of shape (participants, occurrences, cue_offsets).
        Scores select the diagonal where transition lag equals minus cue offset.
        Offset zero and offsets outside the lag window are undefined.
    """
    cue_offsets = np.arange(-max_offset, max_offset + 1)
    lag_labels = np.arange(-max_lag, max_lag + 1)
    access = np.full((*crp.shape[:2], cue_offsets.size), np.nan)

    for offset_index, offset in enumerate(cue_offsets):
        if offset == 0:
            continue
        lag_match = np.where(lag_labels == -offset)[0]
        if lag_match.size:
            source_offset_index = offset + max_offset
            access[:, :, offset_index] = crp[:, :, source_offset_index, lag_match[0]]

    return cue_offsets, access


def _select_cue_offset(
    crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    cue_offset: int,
    max_offset: int = 5,
) -> Float[np.ndarray, " subjects occurrences lags"]:
    """Select one cue-offset CRP curve from subject-level arrays."""
    offset_index = cue_offset + max_offset
    return crp[:, :, offset_index, :]


def _masked_cmap(cmap: str = "viridis") -> Colormap:
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
    crp: Float[np.ndarray, " *batch cue_offsets lags"],
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[
    Integer[np.ndarray, " offsets"],
    Float[np.ndarray, " *batch next_offsets previous_offsets"],
]:
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
    max_offset: int | None = None,
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

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

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

        for rep_idx, selected_repetition_index in enumerate(repetition_indices):
            repetition_subject_values = cue_values[:, selected_repetition_index, :]
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

    format_analysis_axes(
        axis, "Transition Lag", "Conditional Resp. Prob.", contrast_name
    )
    return axis


def plot_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: int | None = None,
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
    """Plot the access-to-repeater diagonal across cue offsets.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
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

    for data_index, data in enumerate(datasets):
        subject_values = subject_rep_cue_crp(
            data,
            trial_masks[data_index],
            min_lag=min_lag,
            max_lag=max_lag,
            max_offset=max_offset,
            size=size,
        )
        cue_offsets, access_values = rep_cue_access(
            subject_values, max_offset=max_offset, max_lag=max_lag
        )

        for rep_idx, selected_repetition_index in enumerate(repetition_indices):
            repetition_subject_values = access_values[:, selected_repetition_index, :]
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
                cue_offsets,
                repetition_subject_values,
                label,
                color,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )

    format_analysis_axes(axis, "Cue Offset", "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_first_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: int | None = None,
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
    """Convenience function to plot first-presentation cue access CRP.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_second_rep_cue_access_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: int | None = None,
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
    """Convenience function to plot second-presentation cue access CRP.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_rep_cue_crp_surface(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    max_lag: int = 5,
    max_offset: int | None = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: int = 0,
    axis: Axes | None = None,
    vmin: Float | None = None,
    vmax: Float | None = None,
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
        extent=(
            -max_lag - 0.5,
            max_lag + 0.5,
            -max_offset - 0.5,
            max_offset + 0.5,
        ),
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
    max_offset: int | None = None,
    min_lag: int = 4,
    size: int = 2,
    repetition_index: int = 0,
    axis: Axes | None = None,
    vmin: Float | None = None,
    vmax: Float | None = None,
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
        extent=(
            -max_offset - 0.5,
            max_offset + 0.5,
            -max_offset - 0.5,
            max_offset + 0.5,
        ),
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


def _format_offset_band(cue_offsets: Iterable[int]) -> str:
    """Return a compact label for an offset band."""
    return ",".join(f"{offset:+d}" for offset in cue_offsets)


def rep_cue_access_band(
    crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> tuple[
    Integer[np.ndarray, " selected_offsets"], Float[np.ndarray, " subjects occurrences"]
]:
    """Average access-to-repeater probabilities across selected cue offsets.

    Parameters
    ----------
    crp : array of shape (participants, occurrences, cue_offsets, lags)
        Participant cue-conditioned probabilities. Offset and lag axes run from
        their negative limit through zero to their positive limit.
    cue_offsets : sequence of int, optional
        Nonzero offsets to average, default (-1, 1); each must be in the offset window.
    max_offset, max_lag : int, optional
        Inclusive windows represented by the supplied array, default five each.

    Returns
    -------
    tuple of arrays
        Selected offsets and scores of shape (participants, occurrences).
        Average finite diagonal values with equal offset weight within a participant.
        No available offsets gives NaN; this does not pool transition opportunities.
    """
    cue_offsets = tuple(cue_offsets)

    all_offsets, access = rep_cue_access(crp, max_offset=max_offset, max_lag=max_lag)
    offset_indices = []
    for cue_offset in cue_offsets:
        matches = np.where(all_offsets == cue_offset)[0]
        offset_indices.append(matches[0])

    selected = access[:, :, offset_indices]
    valid = np.isfinite(selected)
    counts = valid.sum(axis=2)
    summed = np.where(valid, selected, 0).sum(axis=2)
    band = np.full(counts.shape, np.nan)
    band[counts > 0] = summed[counts > 0] / counts[counts > 0]
    return np.asarray(cue_offsets), band


def test_rep_cue_access_vs_control(
    observed_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    control_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    max_offset: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare access-to-repeater probabilities between observed and control data.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    max_offset : int, optional
        Inclusive cue-offset window represented by the supplied arrays.
    max_lag : int, optional
        Inclusive transition-lag window; labels run from -max_lag to +max_lag.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. The index levels are occurrence (one-based) and comparison.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted. Cue access selects transition lag equal to minus cue offset.
    """
    cue_offsets, observed = rep_cue_access(
        observed_crp, max_offset=max_offset, max_lag=max_lag
    )
    _, control = rep_cue_access(control_crp, max_offset=max_offset, max_lag=max_lag)
    size = observed.shape[1]
    results = []

    for rep_idx in range(size):
        results.append(
            compare_paired(
                observed[:, rep_idx, :], control[:, rep_idx, :], labels=cue_offsets
            )
        )

    return pd.concat(results, keys=range(1, size + 1), names=["occurrence"])


def test_rep_cue_access_band_vs_control(
    observed_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    control_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare mean access probabilities across a selected cue-offset band.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    cue_offsets : sequence of int, optional
        Offsets whose finite access probabilities are averaged within each participant.
    max_offset : int, optional
        Inclusive cue-offset window represented by the supplied arrays.
    max_lag : int, optional
        Inclusive transition-lag window; labels run from -max_lag to +max_lag.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. The index levels are occurrence (one-based) and comparison.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted. Cue access selects transition lag equal to minus cue offset.
    """
    selected_offsets, observed = rep_cue_access_band(
        observed_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    _, control = rep_cue_access_band(
        control_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    labels = np.array([_format_offset_band(selected_offsets)])
    size = observed.shape[1]
    results = []

    for rep_idx in range(size):
        results.append(
            compare_paired(
                observed[:, rep_idx, None], control[:, rep_idx, None], labels=labels
            )
        )

    return pd.concat(results, keys=range(1, size + 1), names=["occurrence"])


def test_first_second_bias(
    observed_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    control_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    cue_offset: int,
    max_offset: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare first-minus-second occurrence scores between observed and control data.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    cue_offset : int
        Prior-item offset from the repeated presentation to select.
    max_offset : int, optional
        Inclusive cue-offset window represented by the supplied arrays.
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
    are unadjusted. Cue access selects transition lag equal to minus cue offset.
    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    observed = _select_cue_offset(observed_crp, cue_offset, max_offset)
    control = _select_cue_offset(control_crp, cue_offset, max_offset)

    observed_diff = observed[:, 0, :] - observed[:, 1, :]
    control_diff = control[:, 0, :] - control[:, 1, :]
    return compare_paired(observed_diff, control_diff, labels=lag_labels)


def test_first_second_access_bias(
    observed_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    control_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    max_offset: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare first-minus-second access probabilities at each cue offset.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    max_offset : int, optional
        Inclusive cue-offset window represented by the supplied arrays.
    max_lag : int, optional
        Inclusive transition-lag window; labels run from -max_lag to +max_lag.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. Rows are labeled by cue offset.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted. Cue access selects transition lag equal to minus cue offset.
    """
    cue_offsets, observed = rep_cue_access(
        observed_crp, max_offset=max_offset, max_lag=max_lag
    )
    _, control = rep_cue_access(control_crp, max_offset=max_offset, max_lag=max_lag)

    observed_diff = observed[:, 0, :] - observed[:, 1, :]
    control_diff = control[:, 0, :] - control[:, 1, :]
    return compare_paired(
        observed_diff,
        control_diff,
        labels=cue_offsets,
    )


def test_first_second_access_band_bias(
    observed_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    control_crp: Float[np.ndarray, " subjects occurrences cue_offsets lags"],
    cue_offsets: Sequence[int] = (-1, 1),
    max_offset: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compare first-minus-second access probabilities averaged over cue offsets.

    Parameters
    ----------
    observed_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    control_crp : array
        Scores with axes participant, occurrence, cue offset, transition lag. Participant rows must align across inputs.
    cue_offsets : sequence of int, optional
        Offsets whose finite access probabilities are averaged within each participant.
    max_offset : int, optional
        Inclusive cue-offset window represented by the supplied arrays.
    max_lag : int, optional
        Inclusive transition-lag window; labels run from -max_lag to +max_lag.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. The row label lists the selected cue offsets.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted. Cue access selects transition lag equal to minus cue offset.
    """
    selected_offsets, observed = rep_cue_access_band(
        observed_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )
    _, control = rep_cue_access_band(
        control_crp,
        cue_offsets=cue_offsets,
        max_offset=max_offset,
        max_lag=max_lag,
    )

    observed_diff = observed[:, 0] - observed[:, 1]
    control_diff = control[:, 0] - control[:, 1]
    return compare_paired(
        observed_diff[:, None],
        control_diff[:, None],
        labels=[_format_offset_band(selected_offsets)],
    )
