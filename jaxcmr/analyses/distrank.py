"""Proximity of each recalled item to the previous recall in a distance matrix.

For each transition, the previous item's row of the matrix supplies a
distance to every available study item. The chosen distance is ranked among
those alternatives. The closest scores 1 and the farthest 0 when distances
are distinct. Tied items share their mean rank. Uniform choice among the
available items gives an expected score of 0.5.

Use ``distrank`` for one mean rank per trial and ``subject_distrank`` for the
mean of the defined trial scores per participant. Transitions with fewer
than two available items contribute no rank. The global item identifiers
in ``pres_itemids`` are one-based, so item 1 corresponds to the first row
and column of the matrix.
"""

__all__ = [
    "DistanceRankTabulation",
    "distrank",
    "percentile_rank",
    "plot_distrank",
    "subject_distrank",
    "tabulate_trial",
]

from collections.abc import Sequence
from typing import cast

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import _require_pres_itemids, apply_by_subject

from ..math import cosine_similarity_matrix
from ..plotting import plot_mean_intervals, prepare_plot_inputs
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ._transitions import masked_midrank
from .statistics import mean_interval

# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------


def percentile_rank(
    target: Float[Array, ""],
    pool: Float[Array, " candidates"],
) -> Float[Array, ""]:
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
    return masked_midrank(target, pool, pool != jnp.inf)


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
    ) -> None:
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
) -> Float[Array, ""]:
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

    This numerical operation is JIT-compatible and requires aligned integer
    item IDs within matrix coverage. Supply a square matrix with finite distances
    for the within-list alternatives used by the analysis. Input alignment and
    distance suitability are the caller's responsibility.

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
        dataset["recalls"], _require_pres_itemids(dataset), distance_matrix
    )


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


def subject_distrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    distance_matrix: Float[Array, " item_count item_count"],
) -> Float[np.ndarray, " subjects"]:
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    features: Float[Array, "word_count features_count"] | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    distance_matrix: Float[Array, " item_count item_count"] | None = None,
    confidence_level: float = 0.95,
    bootstrap_method: str = "BCa",
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot subject-wise distance-rank factors as points with error bars.

    Supply features or a distance matrix. When both are supplied, the distance
    matrix takes precedence.

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

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

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

    if distance_matrix is None:
        # The caller supplies features when no distance matrix is given.
        feature_matrix = cast(Float[Array, " word_count features_count"], features)
        distances = 1 - cosine_similarity_matrix(feature_matrix)
    else:
        distances = jnp.asarray(distance_matrix)

    intervals = [
        mean_interval(
            subject_distrank(dataset, mask, distances),
            confidence_level=confidence_level, method=bootstrap_method,
            n_resamples=n_resamples, seed=bootstrap_seed,
        )
        for dataset, mask in zip(datasets, trial_masks, strict=True)
    ]
    return plot_mean_intervals(
        axis, intervals, labels, ylabel="Organization Score",
        color_cycle=color_cycle, contrast_name=contrast_name,
    )
