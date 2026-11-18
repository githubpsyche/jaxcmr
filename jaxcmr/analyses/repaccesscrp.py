"""Recall transitions into an available repeated item.

In the list A, B, X, C, D, E, F, G, X, H, suppose B has just been recalled
and X is still available. B lies one position before the first X and seven
before the second. This gives one opportunity at offset −1 on the first
curve and one at −7 on the second. If X comes next, both receive an actual
count at those offsets.

``repaccesscrp`` pools actual counts and opportunities across trials. Its
horizontal axis locates the previous recall relative to a presentation of
the repeated item. ``repcrp`` answers the complementary question about the
item recalled *after* X.
"""

__all__ = [
    "RepAccessCRPTabulation",
    "plot_rep_access_crp",
    "repaccesscrp",
    "subject_rep_access_crp",
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
from ._transitions import remove_positions
from .statistics import compare_paired


class RepAccessCRPTabulation(Pytree):
    """Tabulate forward access to repeated items.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Presented item indices (1-indexed; 0 = pad).
    first_recall : Int_
        First recalled item.
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
    ) -> None:
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_indices = jnp.arange(self.list_length, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
            self.all_positions,
        )

        first_positions = self.item_study_positions[:, 0]
        second_positions = self.item_study_positions[:, 1]
        self.spaced_repeaters = (second_positions - first_positions) > min_lag

        self.actual_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)
        self.avail_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Update the study positions available to retrieve after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return remove_positions(self.avail_recalls, study_positions)

    def is_repeated_item_available(self, item_index: Int_) -> Bool[Array, ""]:
        """Return whether the repeated item is still available."""
        first_pos = self.item_study_positions[item_index, 0]
        return lax.cond(
            first_pos > 0,
            lambda: self.avail_recalls[first_pos - 1],
            lambda: jnp.array(False),
        )

    def lags_from_previous_to_target(
        self, target_pos: Int_, valid_target: Bool[Array, ""]
    ) -> Bool[Array, " lags"]:
        """Return lag bins from previous recall positions to ``target_pos``."""

        def lags_from_previous_position(
            previous_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                valid_target & (target_pos > 0) & (previous_position > 0),
                lambda: self.base_lags.at[
                    previous_position - target_pos + self.lag_range
                ].add(1),
                lambda: self.base_lags,
            )

        return (
            lax.map(lags_from_previous_position, self.previous_positions)
            .sum(0)
            .astype(bool)
        )

    def tabulate_item_lags(
        self, item_index: Int_, recall: Int_
    ) -> tuple[Integer[Array, " size lags"], Integer[Array, " size lags"]]:
        """Return actual and available access lags for one target item."""
        positions = self.item_study_positions[item_index]
        is_first_occurrence = (item_index + 1) == positions[0]
        valid_target = (
            self.spaced_repeaters[item_index]
            & self.is_repeated_item_available(item_index)
            & is_first_occurrence
        )
        current_is_target = jnp.any(recall == positions)

        avail = lax.map(
            lambda target_pos: self.lags_from_previous_to_target(
                target_pos, valid_target
            ),
            positions,
        )
        actual = avail & current_is_target
        return actual.astype(int), avail.astype(int)

    def tabulate_lags(
        self, recall: Int_
    ) -> tuple[Integer[Array, " size lags"], Integer[Array, " size lags"]]:
        """Return cumulative actual and available access lags."""
        actual, avail = lax.map(
            lambda item_index: self.tabulate_item_lags(item_index, recall),
            self.item_indices,
        )
        return self.actual_lags + actual.sum(0), self.avail_lags + avail.sum(0)

    def is_valid_recall(self, recall: Int_) -> Bool[Array, ""]:
        """Return ``True`` when recall positions have not been retrieved yet."""
        recall_study_positions = self.item_study_positions[recall - 1]
        is_valid_study_position = recall_study_positions != 0
        is_available_study_position = self.avail_recalls[recall_study_positions - 1]
        return jnp.any(is_valid_study_position & is_available_study_position)

    def conditional_tabulate(self, recall: Int_) -> "RepAccessCRPTabulation":
        """Tabulate lags when recall is nonzero and still available."""
        actual_lags, avail_lags = self.tabulate_lags(recall)
        return self.replace(
            previous_positions=self.item_study_positions[recall - 1],
            avail_recalls=self.available_recalls_after(recall),
            actual_lags=actual_lags,
            avail_lags=avail_lags,
        )

    def tabulate(self, recall: Int_) -> "RepAccessCRPTabulation":
        """Tabulate forward access probabilities for a recall event."""
        return lax.cond(
            recall,
            lambda: lax.cond(
                self.is_valid_recall(recall),
                lambda: self.conditional_tabulate(recall),
                lambda: self,
            ),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    size: int = 2,
) -> tuple[Float[Array, " size lags"], Float[Array, " size lags"]]:
    """Tabulate observed and available access lags for a trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events for a single trial.
    presentation : Integer[Array, " study_events"]
        Study events for the trial.
    min_lag : int, optional
        Minimum spacing between repeated occurrences.
    size : int, optional
        Maximum positions an item can occupy.

    Returns
    -------
    tuple of Float[Array, " size lags"]
        Actual and available lag tabulations.

    """
    has_recall = jnp.any(trial > 0)
    first_index = jnp.argmax(trial > 0)

    def tabulate_nonempty() -> tuple[
        Integer[Array, " occurrences lags"], Integer[Array, " occurrences lags"]
    ]:
        init = RepAccessCRPTabulation(presentation, trial[first_index], min_lag, size)
        later_trial = jnp.where(jnp.arange(trial.size) > first_index, trial, 0)
        tab = lax.fori_loop(
            0, trial.size, lambda i, t: t.tabulate(later_trial[i]), init
        )
        return tab.actual_lags, tab.avail_lags

    def tabulate_empty() -> tuple[
        Integer[Array, " occurrences lags"], Integer[Array, " occurrences lags"]
    ]:
        lag_range = presentation.size - 1
        empty_lags = jnp.zeros((size, lag_range * 2 + 1), dtype=int)
        return empty_lags, empty_lags

    return lax.cond(has_recall, tabulate_nonempty, tabulate_empty)


def repaccesscrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, " size lags"]:
    """Forward lag-CRP for access to repeated items.

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
    Float[Array, " size lags"]
        CRP of shape ``(size, 2*L-1)`` per repetition index.

    """
    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentations, min_lag, size
    )
    return actual.sum(0) / possible.sum(0)


def plot_rep_access_crp(
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
    """Plot forward repetition access lag-CRP with CIs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to plot.
    min_lag : int, optional
        Minimum separation between repeated presentations.
    size : int, optional
        Maximum study positions per item.
    repetition_index : int or None, optional
        Specific repetition index to plot.
    color_cycle : list[str] or None, optional
        Colors for plotting each dataset.
    labels : Sequence[str] or None, optional
        Labels for repetition-index lines.
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
        Axes with the access CRP plot.
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
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(repaccesscrp, static_argnames=("min_lag", "size")),
            min_lag=min_lag,
            size=size,
        )

        for rep_idx, selected_repetition_index in enumerate(repetition_indices):
            repetition_subject_values = jnp.vstack(
                [each[selected_repetition_index] for each in subject_values]
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

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


def subject_rep_access_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> Float[np.ndarray, "subjects occurrences lags"]:
    """Compute subject-level forward repetition access CRP.

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

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size, 2*max_lag+1)``.

    """
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(repaccesscrp, static_argnames=("min_lag", "size")),
        min_lag=min_lag,
        size=size,
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
