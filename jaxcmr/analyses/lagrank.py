"""Lag-Rank Temporal Factor Score.

Quantifies temporal organization in free recall by computing the
percentile rank of each transition's absolute lag among all available
absolute lags. A score of 0.5 indicates chance; scores above 0.5
reflect a temporal contiguity effect. Supports unique-item lists
(``SimpleTabulation``) and lists with repeated items (``Tabulation``).

Notes
-----
**Temporal factor scores** (Polyn et al., 2009) offer a summary scalar
for temporal organization that is insensitive to list length and output
position, unlike raw lag-CRP curves. The per-subject mean factor can
be tested against 0.5 with a one-sample t-test.

**Percentile ranking**: for *n* available items, the rank of the chosen
item equals ``(n - rank_position) / (n - 1)`` where ``rank_position``
uses midpoint tie-breaking. Transitions with only one available item
(no choice) are excluded from the mean.

**SimpleTabulation** (no repeats): single ``previous_item`` position,
boolean ``avail_items`` mask, item-level ranking.

**Tabulation** (with repeats): multi-position ``previous_positions``
and boolean lag-bin ranking, mirroring the CRP repeat-item convention.
Available lags and actual lags are computed as boolean unions across
study positions to avoid multiplicity inflation.

**JAX compilation**: all public functions are side-effect-free and
JIT-safe. Use ``jit(lagrank, static_argnames=("size",))``.

"""

__all__ = [
    "percentile_rank",
    "SimpleTabulation",
    "simple_tabulate_trial",
    "simple_lagrank",
    "Tabulation",
    "tabulate_trial",
    "lagrank",
    "subject_lagrank",
    "LagRankTestResult",
    "LagRankComparisonResult",
    "test_lagrank",
    "test_lagrank_vs_comparison",
    "plot_lagrank",
]

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy import stats
from scipy.stats import bootstrap
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import init_plot, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from .crp import set_false_at_index


# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------


def percentile_rank(
    target: Integer[Array, ""],
    pool: Integer[Array, " candidates"],
) -> Float:
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
    valid = pool >= 0
    n = jnp.sum(valid)
    n_smaller = jnp.sum(valid & (pool < target))
    n_equal = jnp.sum(valid & (pool == target))
    return jnp.where(
        n > 1,
        (n - 1 - n_smaller - 0.5 * (n_equal - 1)) / (n - 1),
        jnp.float32(jnp.nan),
    )


# ---------------------------------------------------------------------------
# SimpleTabulation — unique-item lists
# ---------------------------------------------------------------------------


class SimpleTabulation(Pytree):
    """Lag-rank tabulator for lists without repeated items."""

    def __init__(self, list_length: int, first_recall: Int_):
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
        avail_bins = self.base_lags.at[all_abs].add(
            self.avail_items.astype(int)
        ).astype(bool)

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
    ):
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.list_length, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
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
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Absolute-lag bins reachable from *pos* to available items."""
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[jnp.abs(self.all_positions - pos)].add(
                self.avail_recalls
            ),
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

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(recall_pos - prev)].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def should_tabulate(self, recall: Int_) -> Bool:
        """Return True if the recall has at least one available study position."""

        def _for_nonzero():
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
            lax.map(self.lags_from_previous, recall_study_positions)
            .sum(0)
            .astype(bool)
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

        new_rank_sum = jnp.where(
            has_choice, self.rank_sum + mean_rank, self.rank_sum
        )
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
) -> Float:
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


def lagrank(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " trials"]:
    """Compute per-trial lag-rank factors with support for repeated items.

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
) -> np.ndarray:
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
        dataset, trial_mask, jit(lagrank, static_argnames=("size",)), size=size
    )
    return np.array([float(jnp.nanmean(v)) for v in subject_values])


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


@dataclass
class LagRankTestResult:
    """Results from a one-sample lag-rank test against chance."""

    n: int
    mean_factor: float
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float

    def __str__(self) -> str:
        lines = [
            f"N={self.n}",
            f"Mean factor: {self.mean_factor:.4f}",
            f"t-stat: {self.t_stat:.3f} p={self.t_pval:.4f}",
            f"W-stat: {self.w_stat:.1f} p={self.w_pval:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class LagRankComparisonResult:
    """Results from a paired lag-rank comparison."""

    n: int
    mean_observed: float
    mean_comparison: float
    mean_diff: float
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float

    def __str__(self) -> str:
        lines = [
            f"N={self.n}",
            f"Mean factor (observed): {self.mean_observed:.4f}",
            f"Mean factor (comparison): {self.mean_comparison:.4f}",
            f"Mean difference: {self.mean_diff:.4f}",
            f"t-stat: {self.t_stat:.3f} p={self.t_pval:.4f}",
            f"W-stat: {self.w_stat:.1f} p={self.w_pval:.4f}",
        ]
        return "\n".join(lines)


def test_lagrank(
    subject_factors: np.ndarray,
) -> LagRankTestResult:
    """Test whether lag-rank factors differ from chance (0.5).

    Parameters
    ----------
    subject_factors : np.ndarray
        Per-subject mean temporal factors.

    Returns
    -------
    LagRankTestResult
        One-sample t-test and Wilcoxon results.

    """
    valid = np.isfinite(subject_factors)
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_1samp(subject_factors, 0.5, nan_policy="omit")
    if n > 10:
        w_stat, w_pval = stats.wilcoxon(
            subject_factors[valid] - 0.5, alternative="two-sided"
        )
    else:
        w_stat, w_pval = np.nan, np.nan

    return LagRankTestResult(
        n=n,
        mean_factor=float(np.nanmean(subject_factors)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


def test_lagrank_vs_comparison(
    observed: np.ndarray,
    comparison: np.ndarray,
) -> LagRankComparisonResult:
    """Test whether observed and comparison lag-rank factors differ.

    Parameters
    ----------
    observed : np.ndarray
        Per-subject factors (observed condition).
    comparison : np.ndarray
        Per-subject factors (comparison condition).

    Returns
    -------
    LagRankComparisonResult
        Paired t-test and Wilcoxon results.

    """
    valid = ~(np.isnan(observed) | np.isnan(comparison))
    n = int(valid.sum())
    t_stat, t_pval = stats.ttest_rel(observed, comparison, nan_policy="omit")
    if n > 10:
        diff = observed[valid] - comparison[valid]
        w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return LagRankComparisonResult(
        n=n,
        mean_observed=float(np.nanmean(observed)),
        mean_comparison=float(np.nanmean(comparison)),
        mean_diff=float(np.nanmean(observed - comparison)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise lag-rank factors as a bar chart.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    color_cycle : list[str] or None
        Colors for each bar.
    labels : Sequence[str] or None
        Labels for each bar.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the lag-rank bar chart.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    x = np.arange(len(datasets))
    for i, data in enumerate(datasets):
        factors = subject_lagrank(data, trial_masks[i], size=size)
        valid = factors[np.isfinite(factors)]
        mean = float(np.nanmean(factors))
        color = color_cycle[i % len(color_cycle)]

        if len(valid) > 1:
            ci = bootstrap(
                (valid,), np.nanmean, confidence_level=confidence_level
            ).confidence_interval
            yerr = [[mean - ci.low], [ci.high - mean]]
        else:
            yerr = [[0], [0]]

        axis.bar(x[i], mean, color=color, label=labels[i], alpha=0.7, width=0.6)
        axis.errorbar(
            x[i], mean, yerr=yerr, fmt="none", color="black", capsize=5
        )

    axis.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    set_plot_labels(axis, "", "Temporal Factor", contrast_name)
    return axis
