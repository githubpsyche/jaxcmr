"""Distance-Rank Semantic Factor Score.

Condenses semantic organization into a single scalar by computing the
percentile rank of each transition's distance among all currently
available item distances. A score of 0.5 indicates chance; scores above
0.5 mean transitions favor closer available items.

Notes
-----
Distance-rank scores parallel lag-rank temporal factor scores, but use
pairwise item distances instead of serial-position lags. The supplied
``distance_matrix`` is indexed by item identifier, following the
Distance-CRP convention that ``pres_itemids - 1`` selects rows and
columns from the matrix.

"""

__all__ = [
    "percentile_rank",
    "DistanceRankTabulation",
    "tabulate_trial",
    "distrank",
    "subject_distrank",
    "DistRankTestResult",
    "DistRankComparisonResult",
    "test_distrank",
    "test_distrank_vs_comparison",
    "plot_distrank",
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
from ..math import cosine_similarity_matrix
from ..plotting import prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------


def percentile_rank(
    target: Float[Array, ""],
    pool: Float[Array, " candidates"],
) -> Float:
    """Compute midpoint percentile rank of *target* in *pool*.

    Parameters
    ----------
    target : Float[Array, ""]
        Distance value to rank.
    pool : Float[Array, " candidates"]
        Candidate distances; ``inf`` entries are treated as unavailable.

    Returns
    -------
    Float
        Rank in [0, 1] where 1.0 = closest in pool, 0.0 = farthest.
        Returns NaN when fewer than 2 available entries.

    """
    valid = pool != jnp.inf
    n = jnp.sum(valid)
    n_smaller = jnp.sum(valid & (pool < target))
    n_equal = jnp.sum(valid & (pool == target))
    return jnp.where(
        n > 1,
        (n - 1 - n_smaller - 0.5 * (n_equal - 1)) / (n - 1),
        jnp.float32(jnp.nan),
    )


# ---------------------------------------------------------------------------
# Trial tabulation
# ---------------------------------------------------------------------------


class DistanceRankTabulation(Pytree):
    """Accumulates distance-rank scores for a single trial."""

    def __init__(
        self,
        availability_mask: Bool[Array, " study_events"],
        first_recall: Int_,
        trial_distances: Float[Array, "study_events study_events"],
    ):
        self.trial_distances = trial_distances
        self.avail_items = availability_mask
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall
        self.rank_sum = jnp.float32(0.0)
        self.transition_count = jnp.int32(0)

    def _update(self, current_item: Int_) -> "DistanceRankTabulation":
        """Rank the transition and update state."""
        distances_from_prev = self.trial_distances[self.previous_item - 1]
        actual_distance = distances_from_prev[current_item - 1]
        pool = jnp.where(self.avail_items, distances_from_prev, jnp.inf)
        rank = percentile_rank(actual_distance, pool)
        has_choice = jnp.sum(pool != jnp.inf) > 1

        return self.replace(
            previous_item=current_item,
            avail_items=self.avail_items.at[current_item - 1].set(False),
            rank_sum=self.rank_sum + jnp.where(has_choice, rank, jnp.float32(0.0)),
            transition_count=self.transition_count + has_choice.astype(jnp.int32),
        )

    def tabulate(self, choice: Int_) -> "DistanceRankTabulation":
        """Tabulate a transition if the choice is non-zero."""
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    distance_matrix: Float[Array, " item_count item_count"],
) -> Float:
    """Tabulate distance-rank factor for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events encoded as study positions.
    present_ids : Integer[Array, " study_item_ids"]
        Item identifiers at each study position.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise distances indexed by item identifier.

    Returns
    -------
    Float
        Mean distance-rank factor for this trial.

    """
    distance_matrix = jnp.asarray(distance_matrix)
    valid = present_ids > 0
    remapped = jnp.where(valid, present_ids - 1, 0)
    trial_distances = distance_matrix[remapped[:, None], remapped[None, :]]
    trial_distances = jnp.where(valid[:, None] & valid[None, :], trial_distances, 0.0)
    init = DistanceRankTabulation(
        availability_mask=valid,
        first_recall=trial[0],
        trial_distances=trial_distances,
    )
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.rank_sum / tab.transition_count


def distrank(
    dataset: RecallDataset,
    distance_matrix: Float[Array, " item_count item_count"],
) -> Float[Array, " trials"]:
    """Compute per-trial distance-rank factors.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemids``.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise item distances.

    Returns
    -------
    Float[Array, " trials"]
        Per-trial semantic distance factor scores.

    """
    return vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"], dataset["pres_itemids"], distance_matrix
    )


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


def subject_distrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    distance_matrix: Float[Array, " item_count item_count"],
) -> np.ndarray:
    """Compute per-subject mean distance-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise item distances.

    Returns
    -------
    np.ndarray
        Per-subject mean semantic distance factor, shape ``(n_subjects,)``.

    """
    subject_values = apply_by_subject(
        dataset, trial_mask, jit(distrank), distance_matrix
    )
    return np.array([float(jnp.nanmean(v)) for v in subject_values])


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


@dataclass
class DistRankTestResult:
    """Results from a one-sample distance-rank test against chance."""

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
class DistRankComparisonResult:
    """Results from a paired distance-rank comparison."""

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


def test_distrank(
    subject_factors: np.ndarray,
) -> DistRankTestResult:
    """Test whether distance-rank factors differ from chance (0.5).

    Parameters
    ----------
    subject_factors : np.ndarray
        Per-subject mean semantic distance factors.

    Returns
    -------
    DistRankTestResult
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

    return DistRankTestResult(
        n=n,
        mean_factor=float(np.nanmean(subject_factors)),
        t_stat=float(t_stat),
        t_pval=float(t_pval),
        w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        w_pval=float(w_pval) if np.isfinite(w_pval) else np.nan,
    )


def test_distrank_vs_comparison(
    observed: np.ndarray,
    comparison: np.ndarray,
) -> DistRankComparisonResult:
    """Test whether observed and comparison distance-rank factors differ.

    Parameters
    ----------
    observed : np.ndarray
        Per-subject factors.
    comparison : np.ndarray
        Per-subject comparison factors.

    Returns
    -------
    DistRankComparisonResult
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

    return DistRankComparisonResult(
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


def plot_distrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    features: Optional[Float[Array, "word_count features_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    distance_matrix: Optional[Float[Array, " item_count item_count"]] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise distance-rank factors as points with error bars.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    features : Float[Array, "word_count features_count"], optional
        Feature matrix whose rows align with vocabulary items.
    color_cycle : list[str] or None
        Colors for each point.
    labels : Sequence[str] or None
        Labels for each point.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    distance_matrix : Float[Array, " item_count item_count"], optional
        Pairwise distance matrix indexed by item identifier.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with distance-rank points and error bars.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    if (features is None) == (distance_matrix is None):
        raise ValueError("Exactly one of features or distance_matrix must be provided.")

    if distance_matrix is None:
        distances = 1 - cosine_similarity_matrix(features)
    else:
        distances = jnp.asarray(distance_matrix)

    x = np.arange(len(datasets))
    for i, data in enumerate(datasets):
        factors = subject_distrank(data, trial_masks[i], distances)
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

        axis.errorbar(
            x[i],
            mean,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor="black",
            capsize=5,
            linestyle="none",
            markersize=6,
            label=labels[i],
            zorder=3,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    axis.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    set_plot_labels(axis, "", "Organization Score", contrast_name)
    return axis
