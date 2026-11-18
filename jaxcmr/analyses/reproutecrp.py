"""Where recall goes after reaching a repeated item from one of its neighbors.

In the list A, B, X, C, D, E, F, G, X, H, consider the recalls B, X, H.
B immediately preceded the first X at study, so it identifies the incoming
route. H lies at lag +7 from the first X and +1 from the second. Both lags
describe the same outgoing response. The route identifies the neighborhood
of the incoming recall while leaving open which occurrence of X came to mind.

``reproutecrp`` keeps the incoming neighborhood and the presentation used
to measure outgoing lag as separate dimensions. The same-neighborhood
curves measure lag from the incoming neighborhood's X. The switch curves
measure it from the other X. Compare those curves to see whether recalls
cluster near one or the other presentation.
"""

__all__ = [
    "RepRouteCRPTabulation",
    "plot_rep_route_crp",
    "plot_reproutecrp_both",
    "plot_reproutecrp_i2i",
    "plot_reproutecrp_i2j",
    "plot_reproutecrp_j2i",
    "plot_reproutecrp_j2j",
    "plot_reproutecrp_same",
    "plot_reproutecrp_switch",
    "reproutecrp",
    "subject_rep_route_crp",
    "tabulate_trial",
    "test_same_switch_bias",
]

from collections.abc import Sequence
from typing import Literal

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

type Direction = Literal["i2i", "i2j", "j2i", "j2j", "same", "switch", "both"]


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
        Include -2 incoming-neighbor offsets when True.

    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        use_lag2: bool = True,
    ) -> None:
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
            lambda i: equivalent_study_positions(i, presentation, size),
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
        return remove_positions(self.avail_recalls, study_positions)

    def route_from_previous(
        self, recall: Int_
    ) -> tuple[Bool[Array, " route"], Integer[Array, " centers"]]:
        """Return incoming route flags for a repeated recall."""
        centers = self.item_study_positions[recall - 1]
        first_pos = centers[0]
        second_pos = centers[1]
        spaced_repeater = (second_pos - first_pos) > self.min_lag

        i_minus1_pos = first_pos - 1
        i_minus2_pos = first_pos - 2
        j_minus1_pos = second_pos - 1
        j_minus2_pos = second_pos - 2

        i_minus1 = (i_minus1_pos > 0) & jnp.any(self.previous_positions == i_minus1_pos)
        i_minus2 = (
            (i_minus2_pos > 0)
            & jnp.any(self.previous_positions == i_minus2_pos)
            & self.use_lag2
        )
        j_minus1 = (j_minus1_pos > 0) & jnp.any(self.previous_positions == j_minus1_pos)
        j_minus2 = (
            (j_minus2_pos > 0)
            & jnp.any(self.previous_positions == j_minus2_pos)
            & self.use_lag2
        )

        routes = jnp.array(
            [
                spaced_repeater & (i_minus1 | i_minus2),
                spaced_repeater & (j_minus1 | j_minus2),
            ]
        )
        route_centers = jnp.where(jnp.any(routes), centers, 0)
        return routes, route_centers

    def lag_from_pending(self, pos: Int_) -> Integer[Array, " route center lags"]:
        """Return lags from pending occurrence centers to ``pos``."""

        def lags_for_route(
            route_index: Integer[Array, ""],
        ) -> Integer[Array, " centers lags"]:
            def lags_from_center(
                center_index: Integer[Array, ""],
            ) -> Integer[Array, " lags"]:
                center = self.pending_centers[center_index]

                def lag_from_center() -> Integer[Array, " lags"]:
                    return self.base_lags.at[pos - center + self.lag_range].add(1)

                return lax.cond(
                    self.pending_routes[route_index] & (center > 0) & (pos > 0),
                    lag_from_center,
                    lambda: self.base_lags,
                )

            return lax.map(lags_from_center, self.center_indices)

        return lax.map(lags_for_route, self.route_indices)

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

        def lag_from_center() -> Integer[Array, " lags"]:
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
    def should_tabulate(self) -> Bool[Array, ""]:
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

    def is_valid_recall(self, recall: Int_) -> Bool[Array, ""]:
        """Return ``True`` when recall positions have not been retrieved yet."""

        def _for_nonzero() -> Bool[Array, ""]:
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
        Include -2 incoming-neighbor offsets when True.
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
        Include -2 incoming-neighbor offsets when True.
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
    actual, possible = tabulate_trials(
        trials, presentations, direction, use_lag2, min_lag
    )
    return actual.sum(0) / possible.sum(0)


def plot_rep_route_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    direction: Direction = "same",
    use_lag2: bool = True,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot route-conditioned repetition lag-CRP with CIs.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
                    plot_mean_curve(
                        axis,
                        lag_interval,
                        subject_values[:, route_idx, center_idx],
                        label,
                        color,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        bootstrap_seed=bootstrap_seed,
                    )
        else:
            subject_values = jnp.vstack(subject_values)[
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
    format_analysis_axes(axis, xlabel, "Conditional Resp. Prob.", contrast_name)
    return axis


def plot_reproutecrp_i2i(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for i predecessor-side neighbor -> repeater -> i.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_i2j(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for i predecessor-side neighbor -> repeater -> j.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_j2i(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for j predecessor-side neighbor -> repeater -> i.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_j2j(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for j predecessor-side neighbor -> repeater -> j.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_same(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for same-route transitions.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_switch(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for switch-route transitions.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def plot_reproutecrp_both(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 3,
    min_lag: int = 4,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Return plot configured for all route-conditioned transitions.

    Other Parameters
    ----------------
    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.
    """
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
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def subject_rep_route_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    direction: Direction = "same",
    use_lag2: bool = True,
    min_lag: int = 4,
    max_lag: int = 3,
) -> (
    Float[np.ndarray, " subjects lags"]
    | Float[np.ndarray, " subjects routes centers lags"]
):
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


def test_same_switch_bias(
    observed_same_crp: Float[np.ndarray, " subjects lags"],
    observed_switch_crp: Float[np.ndarray, " subjects lags"],
    control_same_crp: Float[np.ndarray, " subjects lags"],
    control_switch_crp: Float[np.ndarray, " subjects lags"],
    max_lag: int = 3,
) -> pd.DataFrame:
    """Compare same-minus-switch route scores between observed and control data.

    Parameters
    ----------
    observed_same_crp : array
        Scores with axes participant and transition lag for the named route. Participant rows must align across inputs.
    observed_switch_crp : array
        Scores with axes participant and transition lag for the named route. Participant rows must align across inputs.
    control_same_crp : array
        Scores with axes participant and transition lag for the named route. Participant rows must align across inputs.
    control_switch_crp : array
        Scores with axes participant and transition lag for the named route. Participant rows must align across inputs.
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
    observed_bias = observed_same_crp - observed_switch_crp
    control_bias = control_same_crp - control_switch_crp
    return compare_paired(
        observed_bias, control_bias, labels=np.arange(-max_lag, max_lag + 1)
    )
