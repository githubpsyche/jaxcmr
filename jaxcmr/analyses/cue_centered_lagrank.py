"""Cue-centered lag-rank temporal factor score.

Computes a temporal factor score from an external retrieval cue to the
recalled item, rather than from the previously recalled item. Parallels
``cue_centered_crp`` but produces a scalar factor per trial instead of a
CRP curve.

Notes
-----
- Cue identity is supplied per recall event via ``cue_clips``; each cue
  is mapped to its study positions the same way recalls are.
- A ``_should_tabulate`` mask controls which events contribute.
- Absolute lags from all cue positions to all recalled-item positions
  are collapsed into boolean bins, then ranked among available absolute
  lags from the cue positions. A score of 1.0 = nearest neighbor,
  0.0 = furthest.

"""

__all__ = [
    "CueCenteredLagRankTabulation",
    "tabulate_trial",
    "cue_centered_lagrank",
    "subject_cue_centered_lagrank",
    "test_cue_centered_lagrank",
    "plot_cue_centered_lagrank",
]

from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from scipy.stats import bootstrap
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import prepare_plot_inputs, set_plot_labels
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset
from .lagrank import LagRankTestResult, test_lagrank


# ---------------------------------------------------------------------------
# Core Tabulation
# ---------------------------------------------------------------------------


def _set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to False using 1-based indexing."""
    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class CueCenteredLagRankTabulation(Pytree):
    """Lag-rank tabulator for cue-centered transitions."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.list_length, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda item: item_to_study_positions(item, presentation, size),
            self.all_positions,
        )

        self.rank_sum = jnp.float32(0.0)
        self.transition_count = jnp.int32(0)
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)

    def item_positions(self, item: Int_) -> Integer[Array, " size"]:
        """Return study positions for an item identifier."""
        zeros = jnp.zeros_like(self.item_study_positions[0])
        return lax.cond(
            (item > 0) & (item <= self.list_length),
            lambda: self.item_study_positions[item - 1],
            lambda: zeros,
        )

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for study positions of a recall."""
        study_positions = self.item_positions(recall)
        return lax.scan(_set_false_at_index, self.avail_recalls, study_positions)[0]

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when recall maps to an available position."""

        def _for_nonzero():
            recall_study_positions = self.item_positions(recall)
            is_valid = recall_study_positions != 0
            is_available = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid & is_available)

        return lax.cond(recall == 0, lambda: jnp.array(False), _for_nonzero)

    def actual_abs_lags(
        self,
        cue_pos: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> Bool[Array, " lags"]:
        """Absolute-lag bins from a cue position to recall positions."""

        def f(rp):
            return lax.cond(
                (cue_pos * rp) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(rp - cue_pos)].add(1),
            )

        return lax.map(f, recall_positions).sum(0).astype(bool)

    def available_abs_lags(self, cue_pos: Int_) -> Bool[Array, " lags"]:
        """Absolute-lag bins from a cue position to available items."""
        return lax.cond(
            cue_pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[jnp.abs(self.all_positions - cue_pos)].add(
                self.avail_recalls
            ),
        )

    def rank_from_cue(
        self,
        cue_positions: Integer[Array, " size"],
        recall_positions: Integer[Array, " size"],
    ) -> tuple[Float, Bool]:
        """Rank the transition from cue to recall."""
        actual = (
            lax.map(
                lambda cp: self.actual_abs_lags(cp, recall_positions),
                cue_positions,
            )
            .sum(0)
            .astype(bool)
        )
        avail = (
            lax.map(self.available_abs_lags, cue_positions).sum(0).astype(bool)
        )

        cumulative = jnp.cumsum(avail)
        total = cumulative[-1]
        n_smaller = cumulative - avail

        has_choice = total > 1
        denom = jnp.maximum(total - 1, 1).astype(jnp.float32)
        ranks = (total - 1 - n_smaller).astype(jnp.float32) / denom

        chosen = actual.astype(jnp.float32)
        mean_rank = jnp.sum(ranks * chosen) / jnp.maximum(jnp.sum(chosen), 1)

        has_any_cue = jnp.any(cue_positions > 0)
        rank = jnp.where(has_choice & has_any_cue, mean_rank, jnp.float32(0.0))
        valid = has_choice & has_any_cue
        return rank, valid

    def tabulate(
        self, recall: Int_, cue: Int_, should_tabulate: Bool_
    ) -> "CueCenteredLagRankTabulation":
        """Update state and optionally rank the transition."""

        def _update_state() -> "CueCenteredLagRankTabulation":
            new_avail = self.available_recalls_after(recall)
            cue_positions = self.item_positions(cue)
            has_cue = jnp.any(cue_positions != 0)

            def _with_rank() -> "CueCenteredLagRankTabulation":
                recall_positions = self.item_positions(recall)
                rank, valid = self.rank_from_cue(cue_positions, recall_positions)
                return self.replace(
                    avail_recalls=new_avail,
                    rank_sum=self.rank_sum + rank,
                    transition_count=self.transition_count + valid.astype(jnp.int32),
                )

            def _without_rank() -> "CueCenteredLagRankTabulation":
                return self.replace(avail_recalls=new_avail)

            should_count = should_tabulate & has_cue
            return lax.cond(should_count, _with_rank, _without_rank)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


# ---------------------------------------------------------------------------
# Trial and dataset functions
# ---------------------------------------------------------------------------


def tabulate_trial(
    recalls: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    cues: Integer[Array, " recall_events"],
    should_tabulate: Bool[Array, " recall_events"],
    size: int = 3,
) -> tuple[Float, Integer]:
    """Tabulate cue-centered lag-rank for a single trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_events"]
        Recall sequence as item identifiers.
    presentation : Integer[Array, " study_events"]
        Study presentation order for the trial.
    cues : Integer[Array, " recall_events"]
        Cue item identifiers aligned to recall events.
    should_tabulate : Bool[Array, " recall_events"]
        Boolean mask; True counts the transition.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    tuple[Float, Integer]
        Rank sum and transition count (scalars).

    """
    init = CueCenteredLagRankTabulation(presentation, size)
    tab = lax.fori_loop(
        0,
        recalls.size,
        lambda i, t: t.tabulate(recalls[i], cues[i], should_tabulate[i]),
        init,
    )
    return tab.rank_sum, tab.transition_count


def cue_centered_lagrank(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " trials"]:
    """Compute per-trial cue-centered lag-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``,
        ``cue_clips``, and ``_should_tabulate``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " trials"]
        Temporal factor per trial; NaN where no transitions qualify.

    """
    should_tab = jnp.asarray(dataset["_should_tabulate"], dtype=bool)
    rank_sums, counts = vmap(tabulate_trial, in_axes=(0, 0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        dataset["cue_clips"],
        should_tab,
        size,
    )
    return rank_sums / counts


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


def subject_cue_centered_lagrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    size: int = 3,
) -> np.ndarray:
    """Compute per-subject mean cue-centered lag-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``,
        ``cue_clips``, and ``_should_tabulate``.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects,)``.

    """
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(cue_centered_lagrank, static_argnames=("size",)),
        size=size,
    )
    return np.array([float(np.nanmean(np.array(v))) for v in subject_values])


# ---------------------------------------------------------------------------
# Statistical test — reuse from lagrank
# ---------------------------------------------------------------------------

test_cue_centered_lagrank = test_lagrank


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cue_centered_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    should_tabulate: (
        Sequence[Bool[Array, " trial_count recall_events"]]
        | Bool[Array, " trial_count recall_events"]
    ),
    size: int = 3,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot cue-centered lag-rank factors as a bar chart.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    should_tabulate : Sequence or array
        Boolean masks aligned to recall events.
    size : int
        Max study positions an item can occupy.
    color_cycle : list[str] or None
        Colors for each bar.
    labels : Sequence[str] or None
        Labels for each dataset.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the bar chart.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]

    if not isinstance(should_tabulate, Sequence):
        should_tabulate = [jnp.array(should_tabulate)]

    x = np.arange(len(datasets))
    width = 0.5

    for d_idx, data in enumerate(datasets):
        data_with_mask = {**data, "_should_tabulate": should_tabulate[d_idx]}
        factors = subject_cue_centered_lagrank(
            data_with_mask, trial_masks[d_idx], size=size
        )
        valid = factors[np.isfinite(factors)]
        mean = float(np.nanmean(factors))
        color = color_cycle[d_idx % len(color_cycle)]

        if len(valid) > 1:
            ci = bootstrap(
                (valid,), np.nanmean, confidence_level=confidence_level
            ).confidence_interval
            yerr = [[mean - ci.low], [ci.high - mean]]
        else:
            yerr = [[0], [0]]

        axis.bar(x[d_idx], mean, width, color=color, label=labels[d_idx], alpha=0.7)
        axis.errorbar(
            x[d_idx], mean, yerr=yerr, fmt="none", color="black", capsize=5
        )

    axis.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    set_plot_labels(axis, "", "Temporal Factor", contrast_name)
    return axis
