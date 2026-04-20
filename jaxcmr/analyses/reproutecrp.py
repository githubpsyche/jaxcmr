"""Route-conditioned repetition lag-conditional response probability.

Measures whether recalling a repeated item after entering from one
occurrence's neighborhood routes the next recall back to that same
neighborhood or to the other occurrence's neighborhood.

Notes
-----
- For a repeated item at study positions i and j, seven direction
  modes are supported:

  - ``i2i``: from i-neighbor to i's neighborhood through the repeater
  - ``i2j``: from i-neighbor to j's neighborhood through the repeater
  - ``j2i``: from j-neighbor to i's neighborhood through the repeater
  - ``j2j``: from j-neighbor to j's neighborhood through the repeater
  - ``same``: union of i2i and j2j
  - ``switch``: union of i2j and j2i
  - ``both``: full 2 x 2 route matrix

- ``use_lag2`` extends the incoming-neighbor definition to include +2
  offsets in addition to the default +1.
- ``test_rep_route_crp_vs_control`` compares observed route CRP
  against a shuffled control.

"""

__all__ = [
    "set_false_at_index",
    "RepRouteCRPTabulation",
    "tabulate_trial",
    "reproutecrp",
    "plot_rep_route_crp",
    "plot_reproutecrp_i2i",
    "plot_reproutecrp_i2j",
    "plot_reproutecrp_j2i",
    "plot_reproutecrp_j2j",
    "plot_reproutecrp_same",
    "plot_reproutecrp_switch",
    "plot_reproutecrp_both",
    "subject_rep_route_crp",
    "test_rep_route_crp_vs_control",
    "test_same_switch_bias",
    "RepRouteCRPTestResult",
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


Direction = Literal["i2i", "i2j", "j2i", "j2j", "same", "switch", "both"]


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


def select_direction(
    actual_lags: Integer[Array, " route center lags"],
    avail_lags: Integer[Array, " route center lags"],
    direction: Direction = "both",
) -> tuple[Integer[Array, "..."], Integer[Array, "..."]]:
    """Select counts for one route direction."""
    if direction == "i2i":
        return actual_lags[0, 0], avail_lags[0, 0]
    if direction == "i2j":
        return actual_lags[0, 1], avail_lags[0, 1]
    if direction == "j2i":
        return actual_lags[1, 0], avail_lags[1, 0]
    if direction == "j2j":
        return actual_lags[1, 1], avail_lags[1, 1]
    if direction == "same":
        actual = actual_lags[0, 0] + actual_lags[1, 1]
        possible = avail_lags[0, 0] + avail_lags[1, 1]
        return actual, possible
    if direction == "switch":
        actual = actual_lags[0, 1] + actual_lags[1, 0]
        possible = avail_lags[0, 1] + avail_lags[1, 0]
        return actual, possible
    return actual_lags, avail_lags


class RepRouteCRPTabulation(Pytree):
    """Tabulate route-conditioned transitions through repeated items.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Presented item indices (1-indexed; 0 = pad).
    first_recall : Int_
        Study position of the first recalled item.
    min_lag : int, optional
        Minimum spacing between repeated presentations.
    use_lag2 : bool, optional
        Include +2 incoming-neighbor offsets when True.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        use_lag2: bool = True,
    ):
        size = 2
        self.min_lag = min_lag
        self.use_lag2 = use_lag2
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.route_indices = jnp.arange(2, dtype=int)
        self.center_indices = jnp.arange(2, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros((2, 2, self.lag_range * 2 + 1), dtype=int)
        self.avail_lags = jnp.zeros((2, 2, self.lag_range * 2 + 1), dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.pending_routes = jnp.zeros(2, dtype=bool)
        self.pending_centers = jnp.zeros(2, dtype=int)
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        "Update the study positions available to retrieve after a transition."
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def route_from_previous(
        self, recall: Int_
    ) -> tuple[Bool[Array, " route"], Integer[Array, " centers"]]:
        """Return incoming route flags for a repeated recall."""
        centers = self.item_study_positions[recall - 1]
        first_pos = centers[0]
        second_pos = centers[1]
        spaced_repeater = (second_pos - first_pos) > self.min_lag

        i_plus1 = jnp.any(self.previous_positions == (first_pos + 1))
        i_plus2 = jnp.any(self.previous_positions == (first_pos + 2)) & self.use_lag2
        j_plus1 = jnp.any(self.previous_positions == (second_pos + 1))
        j_plus2 = jnp.any(self.previous_positions == (second_pos + 2)) & self.use_lag2

        routes = jnp.array(
            [
                spaced_repeater & (i_plus1 | i_plus2),
                spaced_repeater & (j_plus1 | j_plus2),
            ]
        )
        route_centers = jnp.where(jnp.any(routes), centers, 0)
        return routes, route_centers

    def lag_from_pending(self, pos: Int_) -> Integer[Array, " route center lags"]:
        """Return lags from pending occurrence centers to ``pos``."""

        def f(route_idx):
            def g(center_idx):
                center = self.pending_centers[center_idx]

                def lag_from_center():
                    return self.base_lags.at[pos - center + self.lag_range].add(1)

                return lax.cond(
                    self.pending_routes[route_idx] & (center > 0) & (pos > 0),
                    lag_from_center,
                    lambda: self.base_lags,
                )

            return lax.map(g, self.center_indices)

        return lax.map(f, self.route_indices)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        """Return cumulative counts of actual lag transitions."""
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lag_from_pending, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    def available_lags_from_pending_center(
        self, route_idx: Int_, center_idx: Int_
    ) -> Bool[Array, " lags"]:
        """Return available lags from a pending occurrence center."""
        center = self.pending_centers[center_idx]

        def lag_from_center():
            return self.base_lags.at[self.all_positions - center + self.lag_range].add(
                self.avail_recalls
            )

        return lax.cond(
            self.pending_routes[route_idx] & (center > 0),
            lag_from_center,
            lambda: self.base_lags,
        ).astype(bool)

    def available_lags_from_pending_route(
        self, route_idx: Int_
    ) -> Bool[Array, " center lags"]:
        """Return available lags for one pending incoming route."""
        return lax.map(
            lambda center_idx: self.available_lags_from_pending_center(
                route_idx, center_idx
            ),
            self.center_indices,
        )

    def tabulate_available_lags(self) -> Integer[Array, " route center lags"]:
        """Return cumulative counts of available lag transitions."""
        new_lags = lax.map(
            self.available_lags_from_pending_route, self.route_indices
        ).astype(bool)
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self) -> Bool:
        """Return ``True`` when a previous repeater route is pending."""
        return jnp.any(self.pending_routes)

    def conditional_tabulate(self, recall: Int_) -> "RepRouteCRPTabulation":
        """Tabulate lags and update the pending incoming route."""
        new_routes, new_centers = self.route_from_previous(recall)
        return lax.cond(
            self.should_tabulate(),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
                pending_routes=new_routes,
                pending_centers=new_centers,
            ),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                pending_routes=new_routes,
                pending_centers=new_centers,
            ),
        )

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return ``True`` when recall positions have not been retrieved yet."""

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(recall == 0, lambda: jnp.array(False), _for_nonzero)

    def tabulate(self, recall: Int_) -> "RepRouteCRPTabulation":
        """Tabulate actual and available lags from a pending repeater route."""
        return lax.cond(
            self.is_valid_recall(recall),
            lambda: self.conditional_tabulate(recall),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    direction: Direction = "both",
    use_lag2: bool = False,
    min_lag: int = 4,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Tabulate observed and available route-conditioned lags for a trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events for a single trial.
    presentation : Integer[Array, " study_events"]
        Study events for the trial.
    direction : str, optional
        Route-conditioned direction to count.
    use_lag2 : bool, optional
        Include +2 incoming-neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.

    Returns
    -------
    tuple of Float[Array, "..."]
        Actual and available lag tabulations.

    """
    init = RepRouteCRPTabulation(
        presentation, trial[0], min_lag=min_lag, use_lag2=use_lag2
    )
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return select_direction(tab.actual_lags, tab.avail_lags, direction)


def reproutecrp(
    dataset: RecallDataset,
    direction: Direction = "both",
    use_lag2: bool = True,
    min_lag: int = 4,
) -> Float[Array, "..."]:
    """Route-conditioned repetition lag-CRP probabilities per lag.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    direction : str, optional
        Route-conditioned direction to count.
    use_lag2 : bool, optional
        Include +2 incoming-neighbor offsets when True.
    min_lag : int, optional
        Minimum spacing between repeated presentations.

    Returns
    -------
    Float[Array, "..."]
        CRP per lag bin. ``direction="both"`` returns
        ``(2, 2, 2*L-1)``; other directions return ``(2*L-1,)``.

    """

    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    tabulate_trials = vmap(tabulate_trial, in_axes=(0, 0, None, None, None))
    actual, possible = tabulate_trials(trials, presentations, direction, use_lag2, min_lag)
    return actual.sum(0) / possible.sum(0)


def plot_rep_route_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    direction: Direction = "same",
    use_lag2: bool = True,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot route-conditioned repetition lag-CRP with CIs."""
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = int(jnp.max(data["listLength"][trial_masks[data_index]]).item()) - 1

        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(reproutecrp, static_argnames=("direction", "use_lag2", "min_lag")),
            direction,
            use_lag2,
            min_lag,
        )

        if direction == "both":
            subject_values = jnp.stack(subject_values)[
                :, :, :, lag_range - max_lag : lag_range + max_lag + 1
            ]
            route_labels = [["i2i", "i2j"], ["j2i", "j2j"]]
            for route_idx in range(2):
                for center_idx in range(2):
                    color_index = data_index * 4 + route_idx * 2 + center_idx
                    color = color_cycle[color_index % len(color_cycle)]
                    label = (
                        f"{labels[data_index]} {route_labels[route_idx][center_idx]}"
                    ).strip()
                    plot_data(
                        axis,
                        lag_interval,
                        subject_values[:, route_idx, center_idx],
                        label,
                        color,
                        confidence_level=confidence_level,
                    )
        else:
            subject_values = jnp.vstack(subject_values)[
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

    if direction in ("i2i", "j2i"):
        xlabel = "Lag from 1st Presentation"
    elif direction in ("i2j", "j2j"):
        xlabel = "Lag from 2nd Presentation"
    elif direction == "same":
        xlabel = "Lag from Same Presentation"
    elif direction == "switch":
        xlabel = "Lag from Other Presentation"
    else:
        xlabel = "Lag from 1st/2nd Presentation"
    set_plot_labels(axis, xlabel, "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_reproutecrp_i2i(
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
    """Return plot configured for i-neighbor -> repeater -> i transitions."""
    return plot_rep_route_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="i2i",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_reproutecrp_i2j(
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
    """Return plot configured for i-neighbor -> repeater -> j transitions."""
    return plot_rep_route_crp(
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


def plot_reproutecrp_j2i(
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
    """Return plot configured for j-neighbor -> repeater -> i transitions."""
    return plot_rep_route_crp(
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


def plot_reproutecrp_j2j(
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
    """Return plot configured for j-neighbor -> repeater -> j transitions."""
    return plot_rep_route_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="j2j",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_reproutecrp_same(
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
    """Return plot configured for same-route transitions."""
    return plot_rep_route_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="same",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_reproutecrp_switch(
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
    """Return plot configured for switch-route transitions."""
    return plot_rep_route_crp(
        datasets,
        trial_masks,
        max_lag=max_lag,
        min_lag=min_lag,
        direction="switch",
        use_lag2=True,
        color_cycle=color_cycle,
        labels=labels,
        contrast_name=contrast_name,
        axis=axis,
        confidence_level=confidence_level,
    )


def plot_reproutecrp_both(
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
    """Return plot configured for all route-conditioned transitions."""
    return plot_rep_route_crp(
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


def subject_rep_route_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    direction: Direction = "same",
    use_lag2: bool = True,
    min_lag: int = 4,
    max_lag: int = 3,
) -> np.ndarray:
    """Compute subject-level route-conditioned repetition CRP."""
    lag_range = int(np.max(dataset["listLength"][trial_mask])) - 1
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(reproutecrp, static_argnames=("direction", "use_lag2", "min_lag")),
        direction,
        use_lag2,
        min_lag,
    )
    if direction == "both":
        return np.stack([np.array(s)[:, :, lag_slice] for s in subject_values])
    return np.vstack([s[lag_slice] for s in subject_values])


@dataclass
class RepRouteCRPTestResult:
    """Results from a route-conditioned repetition CRP statistical test."""

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


def test_rep_route_crp_vs_control(
    observed_crp: np.ndarray,
    control_crp: np.ndarray,
    max_lag: int = 3,
    direction: str = "same",
) -> RepRouteCRPTestResult:
    """Test observed vs control route-conditioned repetition CRP."""
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

    return RepRouteCRPTestResult(
        lags=lag_labels,
        t_stats=t_stats,
        t_pvals=t_pvals,
        w_stats=w_stats,
        w_pvals=w_pvals,
        mean_diffs=mean_diffs,
        direction=direction,
    )


def test_same_switch_bias(
    observed_same_crp: np.ndarray,
    observed_switch_crp: np.ndarray,
    control_same_crp: np.ndarray,
    control_switch_crp: np.ndarray,
    max_lag: int = 3,
) -> RepRouteCRPTestResult:
    """Test whether same-switch bias differs between observed and control."""
    observed_bias = observed_same_crp - observed_switch_crp
    control_bias = control_same_crp - control_switch_crp
    return test_rep_route_crp_vs_control(
        observed_bias,
        control_bias,
        max_lag=max_lag,
        direction="same-switch",
    )
