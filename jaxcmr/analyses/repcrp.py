"""Repetition conditional response probability.

``RepCRPTabulation`` extends ``crp.Tabulation`` to separately tabulate
lag transitions for each presentation of repeated items, producing one
CRP curve per repetition index.

Notes
-----
- ``avail_lags`` and ``actual_lags`` use 2-D arrays where the first
  dimension is the study position index of the previously recalled
  item and the second is the lag value. The base class collapses
  these into a single dimension; here they remain separate so that
  we can aggregate per-presentation CRPs.
- ``should_tabulate`` restricts tabulation to transitions FROM
  repeated items (all items still update availability tracking).
- ``min_lag`` filters repeated items by minimum spacing between
  their first and second presentations.
- ``tabulate_trial`` returns actual and available lag arrays
  directly rather than the tabulation object.
- ``test_first_second_bias`` tests H0: preference for
  first-presentation neighbors over second-presentation neighbors
  is the same in observed data as in the shuffled control.

"""

__all__ = [
    "set_false_at_index",
    "RepCRPTabulation",
    "tabulate_trial",
    "repcrp",
    "plot_rep_crp",
    "plot_difference_rep_crp",
    "plot_first_rep_crp",
    "plot_second_rep_crp",
    "subject_rep_crp",
    "test_rep_crp_vs_control",
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

from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
from ..repetition import all_study_positions
from ..helpers import apply_by_subject
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def set_false_at_index(vec: Bool[Array, " positions"], i: Int_):
    """Set ``vec[i-1]`` to ``False`` (1-based; 0 is a no-op).

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean vector.
    i : Int_
        1-based index; 0 is a no-op sentinel.

    Returns
    -------
    tuple
        Updated vector and ``None``.

    """
    return lax.cond(i, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None))


class RepCRPTabulation(Pytree):
    """Tabulate lag transitions for repeated items during recall.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Item numbers at each study position (1-indexed).
    first_recall : Int_
        First recalled item number (1-indexed).
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.

    """

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
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)
        self.avail_lags = jnp.zeros((size, self.lag_range * 2 + 1), dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Return study positions available for retrieval after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, pos: Int_) -> Bool[Array, " size positions"]:
        "Identify lags from study positions of the previous item."

        def f(prev):
            return lax.cond(
                (pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[pos - prev + self.lag_range].add(1),
            )

        # modified to not sum over prev_recall axis before returning
        return lax.map(f, self.previous_positions).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulate observed lag transitions for a recall." 
        recall_study_positions = self.item_study_positions[recall - 1]

        # modified to sum over current_recall axis while still separating over prev_recall
        new_lags = lax.map(self.lags_from_previous, recall_study_positions).astype(bool)
        per_lag = jnp.any(new_lags, axis=0)
        return self.actual_lags + per_lag #new_lags.sum(0)

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Return recallable lag transitions from ``pos``."
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Tabulate potential transitions after a recall."

        # modified to not sum over prev_recall axis before summing
        new_lags = lax.map(self.available_lags_from, self.previous_positions)
        return self.avail_lags + new_lags.astype(bool)

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool:
        "Return ``True`` when the prior item was studied twice with spacing > ``min_lag``."
        return (
            len(self.previous_positions) > 1
            and self.previous_positions[-1] - self.previous_positions[-2] > self.min_lag
        )

    def conditional_tabulate(self, recall: Int_) -> "RepCRPTabulation":
        "Tabulate lags only when ``should_tabulate`` is ``True``."
        return lax.cond(
            self.should_tabulate(),
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
        "Update lag counts for a recall index, ignoring ``0`` sentinels."
        return lax.cond(recall, lambda: self.conditional_tabulate(recall), lambda: self)

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
    min_lag : int
        Minimum spacing between item repetitions.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag tabulations.

    """
    init = RepCRPTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags

def repcrp(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, " lags"]:
    """Lag-CRP centered on each presentation of repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum presentations per item.

    Returns
    -------
    Float[Array, " lags"]
        CRP of shape ``(size, 2*L - 1)`` per repetition index.

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
    """Plot repetition Lag-CRP with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
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
        Axes with the repetition Lag-CRP plot.

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
        lag_range = data["pres_itemnos"].shape[1] - 1
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
            )[:, lag_range - max_lag : lag_range + max_lag + 1]

            label = (
                labels_list[data_index]
                if len(repetition_indices) == 1 and data_index < len(labels_list)
                else
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
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot difference between first and second presentation CRPs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
    min_lag : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum presentations per item.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Axes with the difference CRP plot.

    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(datasets, trial_masks, color_cycle, axis)


    if labels is None:
        labels = [""] * len(datasets)



    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = data["pres_itemnos"].shape[1] - 1
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


        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            lag_interval,
            repetition_subject_values,
            str(repetition_index + 1),
            color,
        confidence_level=confidence_level  
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
    max_lag: int = 5,
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    "Convenience function to plot second repetition CRP."
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


def subject_rep_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    max_lag: int = 5,
    size: int = 2,
) -> np.ndarray:
    """Compute subject-level repetition CRP values.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    min_lag : int
        Minimum spacing between item repetitions.
    max_lag : int
        Maximum lag to include in output.
    size : int
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size, 2*max_lag+1)``.

    """
    lag_range = dataset["pres_itemnos"].shape[1] - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset, trial_mask, jit(repcrp, static_argnames=("size",)), min_lag, size
    )
    return np.stack([s[:, lag_slice] for s in subject_values])


@dataclass
class RepCRPTestResult:
    """Results from a repetition CRP statistical test."""

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


def test_rep_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 5,
) -> dict[str, RepCRPTestResult]:
    """Test observed vs control CRP per presentation index.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP, shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level control CRP, same shape.
    max_lag : int
        Maximum lag (used for labeling).

    Returns
    -------
    dict[str, RepCRPTestResult]
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
        results[label] = RepCRPTestResult(
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
) -> RepCRPTestResult:
    """Test whether first-presentation bias differs from control.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP, shape ``(n_subjects, size, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level control CRP, same shape.
    max_lag : int
        Maximum lag (used for labeling).

    Returns
    -------
    RepCRPTestResult
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

        t_stat, t_pval = stats.ttest_rel(obs_d, ctrl_d, nan_policy="omit")
        t_stats[lag_idx] = t_stat
        t_pvals[lag_idx] = t_pval

        valid = ~(np.isnan(obs_d) | np.isnan(ctrl_d))
        if valid.sum() > 10:
            w_stat, w_pval = stats.wilcoxon(diff_of_diff[valid], alternative="two-sided")
        else:
            w_stat, w_pval = np.nan, np.nan
        w_stats[lag_idx] = w_stat
        w_pvals[lag_idx] = w_pval
        mean_diffs[lag_idx] = np.nanmean(diff_of_diff)

    return RepCRPTestResult(
        lags=lag_labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
    )
