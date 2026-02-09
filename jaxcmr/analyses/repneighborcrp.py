"""Repetition-neighbor lag-conditional response probability.

Measures whether recalling the neighbor of one presentation of a
repeated item facilitates transition to the neighborhood of the
other presentation, testing for associative linking across
repetitions.

Notes
-----
- For a repeated item at study positions i and j, three direction
  modes are supported:

  - ``j2i``: from j-neighbor to i's neighborhood
  - ``i2j``: from i-neighbor to j's neighborhood
  - ``both``: union of both directions

- ``use_lag2`` extends the neighbor definition to include +2
  offsets in addition to the default +1.
- ``test_rep_neighbor_crp_vs_control`` compares observed
  neighbor CRP against a shuffled control to test for
  above-chance cross-presentation linking.

"""

__all__ = [
    "set_false_at_index",
    "NeighborCRPTabulation",
    "tabulate_trial",
    "repneighborcrp",
    "plot_rep_neighbor_crp",
    "plot_repneighborcrp_j2i",
    "plot_repneighborcrp_i2j",
    "plot_repneighborcrp_both",
    "subject_rep_neighbor_crp",
    "test_rep_neighbor_crp_vs_control",
    "RepNeighborCRPTestResult",
]

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy import stats
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
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


class NeighborCRPTabulation(Pytree):
    """Tabulate neighbor transitions for repeated items.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Presented item indices (1-indexed; 0 = pad).
    first_recall : Int_
        Study position of the first recalled item.
    direction : {"j2i", "i2j", "both"}, optional
        Which neighbor-transition direction to count.
    use_lag2 : bool, optional
        Include +2 neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        direction: Literal["j2i", "i2j", "both"] = "both",
        use_lag2: bool = True,
    ):
        size = 2
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        # Get first and second study positions
        first_positions = self.item_study_positions[:, 0]
        second_positions = self.item_study_positions[:, 1]

        # construct mask identify repeaters that are sufficiently spaced apart
        spaced_repeaters = (second_positions - first_positions) > min_lag

        # separately identify the first and second positions of just spaced repeaters
        first_mapped = jnp.where(spaced_repeaters, first_positions, 0)
        second_mapped = jnp.where(spaced_repeaters, second_positions, 0)

        # build two 0/1 masks from your args
        use_j2i = direction in ("j2i", "both")
        use_i2j = direction in ("i2j", "both")

        self.ijplus_12_to_ji = (
            jnp.zeros((self.list_length, 4), dtype=int)
            # mapping from second-presentation neighbors j+1, j+2 to i
            .at[second_positions + 1, 0]
            .set(first_mapped * use_j2i)
            .at[second_positions + 2, 1]
            .set(first_mapped * use_j2i * use_lag2)
            # mapping from first-presentation neighbors to j+1/j+2
            .at[first_positions + 1, 2]
            .set(second_mapped * use_i2j)
            .at[first_positions + 2, 3]
            .set(second_mapped * use_i2j * use_lag2)
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Update the study positions available to retrieve after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-from-i-transitions actually made from the previous j+1/2 item
    # and/or lag-from-j-transitions actually made from the previous i+1/2 item
    def lags_from_ij(self, pos: Int_) -> Bool[Array, " positions"]:
        """Return lags from mapped ``i``/``j`` positions to ``pos``."""

        def f(prev):
            def lag_from(ij):
                return self.base_lags.at[pos - ij + self.lag_range].add(1)

            def maybe_lag(ij):
                return lax.cond(
                    (pos * ij) == 0, lambda: self.base_lags, lambda: lag_from(ij)
                )

            return lax.map(maybe_lag, self.ijplus_12_to_ji[prev]).sum(0)

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        """Return cumulative counts of actual lag transitions."""
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_ij, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-from-i-transitions available from the previous j+1/2 item
    def available_lags_from_ij(self, prev_pos: Int_) -> Bool[Array, " lags"]:
        """Return available lag transitions from a mapped ``i`` to all recallable positions."""
        i_values = self.ijplus_12_to_ji[prev_pos]

        def lag_from(ij):
            return self.base_lags.at[self.all_positions - ij + self.lag_range].add(
                self.avail_recalls
            )

        def maybe_lag(ij):
            return lax.cond(ij == 0, lambda: self.base_lags, lambda: lag_from(ij))

        return lax.map(maybe_lag, i_values).sum(0).astype(bool)

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        """Return cumulative counts of available lag transitions."""
        new_lags = (
            lax.map(self.available_lags_from_ij, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool:
        """Return ``True`` when the previous recall enables tabulation."""
        return jnp.any(self.ijplus_12_to_ji[self.previous_positions] > 0)

    def conditional_tabulate(self, recall: Int_) -> "NeighborCRPTabulation":
        """Tabulate lags when the additional condition is met."""
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

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return ``True`` when recall positions have not been retrieved yet."""
        recall_study_positions = self.item_study_positions[recall - 1]
        is_valid_study_position = recall_study_positions != 0
        is_available_study_position = self.avail_recalls[recall_study_positions - 1]
        return jnp.any(is_valid_study_position & is_available_study_position)

    def tabulate(self, recall: Int_) -> "NeighborCRPTabulation":
        """Tabulate actual and available serial lags from the previous item."""
        return lax.cond(
            self.is_valid_recall(recall),
            lambda: self.conditional_tabulate(recall),
            lambda: self,
        )

def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    direction: Literal["j2i", "i2j", "both"] = "j2i",
    use_lag2: bool = False,
    min_lag: int = 4,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate observed and available lags for a trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events for a single trial.
    presentation : Integer[Array, " study_events"]
        Study events for the trial.
    direction : {"j2i", "i2j", "both"}, optional
        Which neighbor-transition direction to count.
    use_lag2 : bool, optional
        Include +2 neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.

    Returns
    -------
    tuple of Float[Array, " lags"]
        Actual and available lag tabulations.

    """
    init = NeighborCRPTabulation(
        presentation, trial[0], min_lag=min_lag, direction=direction, use_lag2=use_lag2
    )
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def repneighborcrp(
    dataset: RecallDataset,
    direction: Literal["j2i", "i2j", "both"] = "both",
    use_lag2: bool = True,
    min_lag: int = 4,
) -> Float[Array, " lags"]:
    """Repetition-neighbor lag-CRP probabilities per lag.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    direction : {"j2i", "i2j", "both"}, optional
        Which neighbor-transition direction to count.
    use_lag2 : bool, optional
        Include +2 neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.

    Returns
    -------
    Float[Array, " lags"]
        CRP of shape ``(2*L-1,)`` per lag bin.

    """

    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    tabulate_trials = vmap(tabulate_trial, in_axes=(0, 0, None, None, None))
    actual, possible = tabulate_trials(trials, presentations, direction, use_lag2, min_lag)
    return actual.sum(0) / possible.sum(0)

def plot_rep_neighbor_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    direction: str = "both",
    use_lag2: bool = True,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,

) -> Axes:
    """Plot repetition-neighbor lag-CRP with CIs.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to display.
    min_lag : int, optional
        Minimum spacing between repeated occurrences.
    direction : str, optional
        Direction of neighbor transitions.
    use_lag2 : bool, optional
        Include +2 neighbor offsets when True.
    color_cycle : list[str] or None, optional
        Colors for each curve.
    labels : Sequence[str] or None, optional
        Legend labels for each dataset.
    contrast_name : str or None, optional
        Legend title.
    axis : Axes or None, optional
        Existing Axes to plot on.
    confidence_level : float, optional
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Axes with the neighbor CRP plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(datasets, trial_masks, color_cycle, axis)


    if labels is None:
        labels = [""] * len(datasets)



    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = int(jnp.max(data["listLength"][trial_masks[data_index]]).item()) - 1

        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(repneighborcrp, static_argnames=("direction", "use_lag2", "min_lag")),
                direction,
                use_lag2,
                min_lag,
            )
        )
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

    # build a dynamic x-axis label
    if direction == "j2i":
        xlabel = "Lag from 1st Presentation"
    elif direction == "i2j":
        xlabel = "Lag from 2nd Presentation"
    else:
        xlabel = "Lag from 1st/2nd Presentation"
    set_plot_labels(axis, xlabel, "Conditional Resp. Prob.", contrast_name)
    return axis


# wrapper functions configuring each direction, always using lag2:

def plot_repneighborcrp_j2i(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Return plot configured for j-neighbor → i transitions."""
    return plot_rep_neighbor_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="j2i",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_repneighborcrp_i2j(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Return plot configured for i-neighbor → j transitions."""
    return plot_rep_neighbor_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="i2j",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_repneighborcrp_both(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Return plot configured for both j-neighbor → i and i-neighbor → j transitions."""
    return plot_rep_neighbor_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="both",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def subject_rep_neighbor_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    direction: Literal["j2i", "i2j", "both"] = "both",
    use_lag2: bool = True,
    min_lag: int = 4,
    max_lag: int = 3,
) -> np.ndarray:
    """Compute subject-level repetition-neighbor CRP.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials to include.
    direction : {"j2i", "i2j", "both"}, optional
        Which neighbor-transition direction to count.
    use_lag2 : bool, optional
        Include +2 neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.
    max_lag : int, optional
        Maximum lag to include in output.

    """
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(repneighborcrp, static_argnames=("direction", "use_lag2", "min_lag")),
        direction,
        use_lag2,
        min_lag,
    )
    return np.vstack([s[lag_slice] for s in subject_values])


@dataclass
class RepNeighborCRPTestResult:
    """Results from a repetition-neighbor CRP statistical test."""

    lags: np.ndarray
    t_stats: np.ndarray
    t_pvals: np.ndarray
    w_stats: np.ndarray
    w_pvals: np.ndarray
    mean_diffs: np.ndarray
    direction: str

    def __str__(self) -> str:
        lines = [
            f"Direction: {self.direction}",
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


def test_rep_neighbor_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 3,
    direction: str = "both",
) -> RepNeighborCRPTestResult:
    """Test observed vs control repetition-neighbor CRP.

    Parameters
    ----------
    observed_crp : np.ndarray
        Subject-level CRP from observed data.
        Shape ``(n_subjects, 2*max_lag+1)``.
    control_crp : np.ndarray
        Subject-level CRP from control data.
        Shape ``(n_subjects, 2*max_lag+1)``.
    max_lag : int, optional
        Maximum lag used for labeling.
    direction : str, optional
        Direction label for results.

    """
    lag_labels = np.arange(-max_lag, max_lag + 1)
    n_lags = len(lag_labels)

    t_stats = np.zeros(n_lags)
    t_pvals = np.zeros(n_lags)
    w_stats = np.zeros(n_lags)
    w_pvals = np.zeros(n_lags)
    mean_diffs = np.zeros(n_lags)

    for lag_idx in range(n_lags):
        obs_col = observed_crp[:, lag_idx]
        ctrl_col = control_crp[:, lag_idx]
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

    return RepNeighborCRPTestResult(
        lags=lag_labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
        direction=direction,
    )
