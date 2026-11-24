"""Temporal proximity of successive recalls relative to the choices still available.

After studying A, B, C, D and recalling B first, the available items A, C
and D lie at absolute lags 1, 1 and 2. The default ``lagrank`` calculation
ranks the two distinct lags. A transition to A or C scores 1, and a
transition to D scores 0. Choosing an item uniformly in this example gives
an expected score of 2/3.

With ``rank_method="item_midrank"``, each available item enters the ranking.
A and C share the mean rank of their tied positions, so their scores are
both 0.75 and D still scores 0. The expected score for uniform item choice
is then 0.5. This method requires distinct study items. The default also
handles repeated items by considering their possible study positions.

``lagrank`` returns one mean score per trial. Transitions need a choice among
at least two lag bins or two items, depending on the method. Trials without
a scored transition return NaN. ``subject_lagrank`` averages the defined
trial scores within each participant.

Use complete lists of the same length. When compiling ``lagrank``, keep
``size`` and ``rank_method`` static.
"""

__all__ = [
    "SimpleTabulation",
    "Tabulation",
    "item_midrank_trial",
    "lagrank",
    "percentile_rank",
    "plot_lagrank",
    "simple_lagrank",
    "simple_tabulate_trial",
    "subject_lagrank",
    "tabulate_trial",
]

from collections.abc import Sequence
from typing import Literal

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..plotting import plot_mean_intervals, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import absolute_lag_counts, masked_midrank, remove_positions
from .statistics import mean_interval

# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------


def percentile_rank(
    target: Integer[Array, ""],
    pool: Integer[Array, " candidates"],
) -> Float[Array, ""]:
    """Compute midpoint percentile rank of *target* in *pool*.

    Parameters
    ----------
    target : Integer[Array, ""]
        Value to rank.
    pool : Integer[Array, " candidates"]
        Candidate values; entries < 0 are treated as missing.

    Returns
    -------
    Float
        Rank in [0, 1] where 1.0 = smallest in pool, 0.0 = largest.
        Returns NaN when fewer than 2 valid entries.

    """
    return masked_midrank(target, pool, pool >= 0)


# ---------------------------------------------------------------------------
# SimpleTabulation — unique-item lists
# ---------------------------------------------------------------------------


class SimpleTabulation(Pytree):
    """Lag-rank tabulator for lists without repeated items."""

    def __init__(self, list_length: int, first_recall: Int_) -> None:
        self.list_length = list_length
        self.all_positions = jnp.arange(1, list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(list_length, dtype=int)
        self.avail_items = jnp.ones(list_length, dtype=bool)
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall
        self.rank_sum = jnp.float32(0.0)
        self.transition_count = jnp.int32(0)

    def _update(self, current_item: Int_) -> "SimpleTabulation":
        """Rank the transition and update state."""
        actual_abs = jnp.abs(current_item - self.previous_item)
        all_abs = jnp.abs(self.all_positions - self.previous_item)

        # Boolean lag bins for available items
        avail_bins = (
            self.base_lags.at[all_abs].add(self.avail_items.astype(int)).astype(bool)
        )

        # Rank actual lag among available bins
        cumulative = jnp.cumsum(avail_bins)
        total = cumulative[-1]
        n_smaller = cumulative - avail_bins

        has_choice = total > 1
        denom = jnp.maximum(total - 1, 1).astype(jnp.float32)
        ranks = (total - 1 - n_smaller).astype(jnp.float32) / denom

        rank = jnp.where(has_choice, ranks[actual_abs], jnp.float32(0.0))
        return self.replace(
            previous_item=current_item,
            avail_items=self.avail_items.at[current_item - 1].set(False),
            rank_sum=self.rank_sum + rank,
            transition_count=self.transition_count + has_choice.astype(jnp.int32),
        )

    def update(self, choice: Int_) -> "SimpleTabulation":
        """Tabulate a transition if the choice is non-zero."""
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def simple_tabulate_trial(
    trial: Integer[Array, " recall_events"], list_length: int
) -> SimpleTabulation:
    """Tabulate lag-rank for a single no-repeat trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in 1..L with 0 pads.
    list_length : int
        Study-list length.

    Returns
    -------
    SimpleTabulation
        Accumulated rank sum and transition count.

    """
    return lax.scan(
        lambda tab, recall: (tab.update(recall), None),
        SimpleTabulation(list_length, trial[0]),
        trial[1:],
    )[0]


def simple_lagrank(
    trials: Integer[Array, "trials recall_events"], list_length: int
) -> Float[Array, " trials"]:
    """Compute per-trial lag-rank factors for unique-item lists.

    Parameters
    ----------
    trials : Integer[Array, "trials recall_events"]
        Serial positions in 1..L with 0 pads.
    list_length : int
        Study-list length.

    Returns
    -------
    Float[Array, " trials"]
        Per-trial temporal factor scores.

    """
    tabs = lax.map(lambda t: simple_tabulate_trial(t, list_length), trials)
    return tabs.rank_sum / tabs.transition_count


# ---------------------------------------------------------------------------
# Tabulation — repeated-item lists
# ---------------------------------------------------------------------------


class Tabulation(Pytree):
    """Lag-rank tabulator supporting repeated items."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ) -> None:
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.list_length, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.rank_sum = jnp.float32(0.0)
        self.transition_count = jnp.int32(0)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for all study positions of a recall."""
        study_positions = self.item_study_positions[recall - 1]
        return remove_positions(self.avail_recalls, study_positions)

    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Absolute-lag bins reachable from *pos* to available items."""
        return absolute_lag_counts(
            pos, self.all_positions, self.avail_recalls, self.all_positions.size - 1
        )

    def available_lags(self) -> Integer[Array, " lags"]:
        """Union of absolute-lag bins from all previous positions."""
        return (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )

    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " lags"]:
        """Absolute-lag bins from previous positions to *recall_pos*."""

        def lags_from_previous_position(
            previous_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                (recall_pos * previous_position) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(recall_pos - previous_position)].add(
                    1
                ),
            )

        return (
            lax.map(lags_from_previous_position, self.previous_positions)
            .sum(0)
            .astype(bool)
        )

    def should_tabulate(self, recall: Int_) -> Bool[Array, ""]:
        """Return True if the recall has at least one available study position."""

        def _for_nonzero() -> Bool[Array, ""]:
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid = recall_study_positions != 0
            is_available = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid & is_available)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def _tabulate(self, recall: Int_) -> "Tabulation":
        """Rank the transition and update state."""
        recall_study_positions = self.item_study_positions[recall - 1]
        actual_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        avail_lags = self.available_lags()

        cumulative = jnp.cumsum(avail_lags)
        total = cumulative[-1]
        n_smaller = cumulative - avail_lags

        has_choice = total > 1
        denom = jnp.maximum(total - 1, 1).astype(jnp.float32)
        ranks = (total - 1 - n_smaller).astype(jnp.float32) / denom

        chosen = actual_lags.astype(jnp.float32)
        mean_rank = jnp.sum(ranks * chosen) / jnp.maximum(jnp.sum(chosen), 1)

        new_rank_sum = jnp.where(has_choice, self.rank_sum + mean_rank, self.rank_sum)
        return self.replace(
            previous_positions=recall_study_positions,
            avail_recalls=self.available_recalls_after(recall),
            rank_sum=new_rank_sum,
            transition_count=self.transition_count + has_choice.astype(jnp.int32),
        )

    def tabulate(self, recall: Int_) -> "Tabulation":
        """Tabulate lag-rank for this transition."""
        return lax.cond(
            self.should_tabulate(recall),
            lambda: self._tabulate(recall),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> Float[Array, ""]:
    """Tabulate lag-rank factor for a single trial with repeats.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in 1..L with 0 pads.
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    size : int
        Max study positions per item.

    Returns
    -------
    Float
        Mean temporal factor for this trial.

    """
    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.rank_sum / tab.transition_count


def item_midrank_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
) -> Float[Array, ""]:
    """Rank available study items with midpoint ties on a unique-item list.

    Recall codes are one-indexed study positions; zeros are padding. Invalid or
    repeated recall events break adjacency. Repeated study identities are not
    supported by this explicitly item-based definition and return NaN.
    """
    positions = jnp.arange(1, presentation.size + 1)
    present = presentation > 0
    duplicate_study = jnp.any(
        (presentation[:, None] == presentation[None, :])
        & present[:, None]
        & present[None, :]
        & ~jnp.eye(presentation.size, dtype=bool)
    )

    def step(
        state: tuple[
            Bool[Array, " study_events"],
            Integer[Array, ""],
            Float[Array, ""],
            Integer[Array, ""],
        ],
        choice: Integer[Array, ""],
    ) -> tuple[
        tuple[
            Bool[Array, " study_events"],
            Integer[Array, ""],
            Float[Array, ""],
            Integer[Array, ""],
        ],
        None,
    ]:
        available, previous, total, count = state
        index = jnp.clip(choice - 1, 0, presentation.size - 1)
        valid = (choice > 0) & (choice <= presentation.size) & available[index]
        pool = jnp.where(available, jnp.abs(positions - previous), -1)
        eligible = valid & (previous > 0) & (jnp.sum(available) > 1)
        rank = percentile_rank(jnp.abs(choice - previous), pool)
        available = available.at[index].set(available[index] & ~valid)
        return (
            available,
            jnp.where(valid, choice, 0),
            total + jnp.where(eligible, rank, 0.0),
            count + eligible,
        ), None

    initial = (present, jnp.array(0), jnp.array(0.0), jnp.array(0))
    (_, _, total, count), _ = lax.scan(step, initial, trial)
    return jnp.where((count > 0) & ~duplicate_study, total / count, jnp.nan)


def lagrank(
    dataset: RecallDataset,
    size: int = 3,
    rank_method: Literal["lag_bins", "item_midrank"] = "lag_bins",
) -> Float[Array, " trials"]:
    """Compute per-trial factors using lag bins (default) or item midpoint ranks.

    ``rank_method="item_midrank"`` requires unique study identities. When using
    JIT, mark both ``size`` and ``rank_method`` as static arguments.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " trials"]
        Per-trial temporal factor scores.

    """
    if rank_method == "item_midrank":
        return vmap(item_midrank_trial)(dataset["recalls"], dataset["pres_itemnos"])
    return vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"], dataset["pres_itemnos"], size
    )


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


def subject_lagrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    size: int = 3,
    rank_method: Literal["lag_bins", "item_midrank"] = "lag_bins",
) -> Float[np.ndarray, " subjects"]:
    """Compute per-subject mean lag-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    np.ndarray
        Per-subject mean temporal factor, shape ``(n_subjects,)``.

    """
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(lagrank, static_argnames=("size", "rank_method")),
        size=size,
        rank_method=rank_method,
    )
    return np.array([float(jnp.nanmean(v)) for v in subject_values])


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    size: int = 3,
    confidence_level: float = 0.95,
    rank_method: Literal["lag_bins", "item_midrank"] = "lag_bins",
    bootstrap_method: str = "BCa",
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot subject-wise lag-rank factors as points with error bars.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    color_cycle : list[str] or None
        Colors for each point.
    labels : Sequence[str] or None
        Labels for each point.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
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
        Matplotlib Axes with lag-rank points and error bars.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    intervals = [
        mean_interval(
            subject_lagrank(dataset, mask, size=size, rank_method=rank_method),
            confidence_level=confidence_level, method=bootstrap_method,
            n_resamples=n_resamples, seed=bootstrap_seed,
        )
        for dataset, mask in zip(datasets, trial_masks, strict=True)
    ]
    return plot_mean_intervals(
        axis, intervals, labels, ylabel="Organization Score",
        color_cycle=color_cycle, contrast_name=contrast_name,
    )
