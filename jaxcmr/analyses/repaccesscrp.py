"""Forward repetition access lag-CRP.

Measures how strongly items at different lags from repeated-item
presentations cue transition to the repeated item. This is the
forward-conditioned access analysis: it asks whether the previous
recall position predicts access to an available repeated item.

"""

__all__ = [
    "set_false_at_index",
    "RepAccessCRPTabulation",
    "tabulate_trial",
    "repaccesscrp",
    "plot_rep_access_crp",
    "subject_rep_access_crp",
    "RepAccessCRPTestResult",
    "test_rep_access_crp_vs_control",
    "test_first_second_bias",
]

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
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
    ):
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_indices = jnp.arange(self.list_length, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
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
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def is_repeated_item_available(self, item_index: Int_) -> Bool:
        """Return whether the repeated item is still available."""
        first_pos = self.item_study_positions[item_index, 0]
        return lax.cond(
            first_pos > 0,
            lambda: self.avail_recalls[first_pos - 1],
            lambda: jnp.array(False),
        )

    def lags_from_previous_to_target(
        self, target_pos: Int_, valid_target: Bool
    ) -> Bool[Array, " lags"]:
        """Return lag bins from previous recall positions to ``target_pos``."""

        def f(prev):
            return lax.cond(
                valid_target & (target_pos > 0) & (prev > 0),
                lambda: self.base_lags.at[
                    prev - target_pos + self.lag_range
                ].add(1),
                lambda: self.base_lags,
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

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

    def is_valid_recall(self, recall: Int_) -> Bool:
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

    def tabulate_nonempty():
        init = RepAccessCRPTabulation(presentation, trial[first_index], min_lag, size)
        later_trial = jnp.where(jnp.arange(trial.size) > first_index, trial, 0)
        tab = lax.fori_loop(0, trial.size, lambda i, t: t.tabulate(later_trial[i]), init)
        return tab.actual_lags, tab.avail_lags

    def tabulate_empty():
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
    repetition_index: Optional[int] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
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

        for rep_idx, repetition_index in enumerate(repetition_indices):
            repetition_subject_values = jnp.vstack(
                [each[repetition_index] for each in subject_values]
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

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


def subject_rep_access_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> np.ndarray:
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


@dataclass
class RepAccessCRPTestResult:
    """Results from a forward repetition access CRP statistical test."""

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


def test_rep_access_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 5,
) -> dict[str, RepAccessCRPTestResult]:
    """Test observed vs control access CRP per index.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    max_lag : int, optional
        Maximum lag used for labeling.

    Returns
    -------
    dict[str, RepAccessCRPTestResult]
        Results keyed by presentation label.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    n_lags = len(lag_labels)
    size = observed_crp.shape[1]
    results = {}

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

        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"
        results[label] = RepAccessCRPTestResult(
            lags=lag_labels,
            t_stats=t_stats,
            t_pvals=t_pvals,
            w_stats=w_stats,
            w_pvals=w_pvals,
            mean_diffs=mean_diffs,
        )

    return results


def test_first_second_bias(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 5,
) -> RepAccessCRPTestResult:
    """Test whether first-presentation access bias differs from control.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, size, 2*max_lag+1)``.
    max_lag : int, optional
        Maximum lag used for labeling.

    Returns
    -------
    RepAccessCRPTestResult
        Test statistics for the difference of differences.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    n_lags = len(lag_labels)

    observed_diff = observed_crp[:, 0, :] - observed_crp[:, 1, :]
    control_diff = control_crp[:, 0, :] - control_crp[:, 1, :]
    effect = observed_diff - control_diff

    t_stats = np.zeros(n_lags)
    t_pvals = np.zeros(n_lags)
    w_stats = np.zeros(n_lags)
    w_pvals = np.zeros(n_lags)
    mean_diffs = np.zeros(n_lags)

    for lag_idx in range(n_lags):
        obs_d = observed_diff[:, lag_idx]
        ctrl_d = control_diff[:, lag_idx]
        diff_of_diff = effect[:, lag_idx]
        valid = ~(np.isnan(obs_d) | np.isnan(ctrl_d))

        if valid.sum() > 1:
            t_stat, t_pval = stats.ttest_rel(obs_d, ctrl_d, nan_policy="omit")
        else:
            t_stat, t_pval = np.nan, np.nan
        t_stats[lag_idx] = t_stat
        t_pvals[lag_idx] = t_pval

        if valid.sum() > 10:
            w_stat, w_pval = stats.wilcoxon(diff_of_diff[valid], alternative="two-sided")
        else:
            w_stat, w_pval = np.nan, np.nan
        w_stats[lag_idx] = w_stat
        w_pvals[lag_idx] = w_pval
        mean_diffs[lag_idx] = np.nanmean(diff_of_diff) if valid.any() else np.nan

    return RepAccessCRPTestResult(
        lags=lag_labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
    )
