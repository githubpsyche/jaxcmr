"""Rank the proximity of a recalled item to an external cue.

Suppose A, B, C, D were studied in order and a cue identifies D. A response
of C lies one study position from that cue. To score it, the analysis ranks
its absolute lag among the distinct absolute lags available from D at that
moment. The closest lag scores 1 and the farthest scores 0 when at least
two different lags are available.

``cue_centered_lagrank`` returns one mean rank per trial and
``subject_cue_centered_lagrank`` averages the defined trial scores within
participants. For the probability of moving in each direction and by each
lag from the cue, use ``cue_centered_crp``.

This experimental analysis has a provisional interface and interpretation.
"""

__all__ = [
    "CueCenteredLagRankTabulation",
    "cue_centered_lagrank",
    "plot_cue_centered_lagrank",
    "subject_cue_centered_lagrank",
    "tabulate_trial",
]

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..analyses._transitions import remove_positions
from ..analyses.statistics import mean_interval
from ..plotting import format_analysis_axes, prepare_plot_inputs
from ..repetition import item_to_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset

# ---------------------------------------------------------------------------
# Core Tabulation
# ---------------------------------------------------------------------------


class CueCenteredLagRankTabulation(Pytree):
    """Lag-rank tabulator for cue-centered transitions."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        size: int = 3,
    ) -> None:
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
        return remove_positions(self.avail_recalls, study_positions)

    def is_valid_recall(self, recall: Int_) -> Bool[Array, ""]:
        """Return True when recall maps to an available position."""

        def _for_nonzero() -> Bool[Array, ""]:
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

        def lags_to_recalled_position(
            recall_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                (cue_pos * recall_position) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(recall_position - cue_pos)].add(1),
            )

        return lax.map(lags_to_recalled_position, recall_positions).sum(0).astype(bool)

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
        avail = lax.map(self.available_abs_lags, cue_positions).sum(0).astype(bool)

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

    should_tab = jnp.asarray(
        cast(Mapping[str, Array], dataset)["_should_tabulate"], dtype=bool
    )
    rank_sums, counts = vmap(tabulate_trial, in_axes=(0, 0, 0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        cast(Mapping[str, Array], dataset)["cue_clips"],
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
) -> Float[np.ndarray, " subjects"]:
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
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot cue-centered lag-rank factors as points with error bars.

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
        Colors for each point.
    labels : Sequence[str] or None
        Labels for each dataset.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
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
        Matplotlib Axes with cue-centered lag-rank points and error bars.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(datasets))]

    if not isinstance(should_tabulate, Sequence):
        should_tabulate = [jnp.array(should_tabulate)]

    x = np.arange(len(datasets))

    for d_idx, data in enumerate(datasets):
        data_with_mask = data.copy()
        data_with_mask["_should_tabulate"] = should_tabulate[d_idx]
        factors = subject_cue_centered_lagrank(
            data_with_mask, trial_masks[d_idx], size=size
        )
        valid = factors[np.isfinite(factors)]
        color = color_cycle[d_idx % len(color_cycle)]

        interval = mean_interval(
            valid,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
        )
        mean = interval["mean"]
        yerr = [[mean - interval["low"]], [interval["high"] - mean]]

        axis.errorbar(
            x[d_idx],
            mean,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor="black",
            capsize=5,
            linestyle="none",
            markersize=6,
            label=labels[d_idx],
            zorder=3,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    axis.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    format_analysis_axes(axis, "", "Organization Score", contrast_name)
    return axis
