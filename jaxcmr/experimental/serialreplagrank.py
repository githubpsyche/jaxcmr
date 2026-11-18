"""Proximity ranks for the first qualifying transition from a repeated item in serial recall.

On the study list A, B, X, C, D, E, F, G, X, H, an opening sequence
A, B, X, H qualifies. X was reached in study order. The destination H lies
close to the second presentation of X and far from the first. The rank for
each presentation compares that distance with the distinct absolute lags
to items still available from that presentation.

``serialreplagrank`` returns scores by trial and presentation number. Once
a transition qualifies, the remaining recalls on that trial contribute no
ranks. The first recall initializes the run, and order checks begin with
the second response. Select trials with a correct first recall if the run
must begin at study position 1. A presentation with no rankable choice has an undefined score.
``subject_serial_rep_lagrank`` averages defined trial scores within participants.

This experimental analysis has a provisional interface and interpretation.
"""

__all__ = [
    "SerialRepLagRankTabulation",
    "plot_serial_rep_lagrank",
    "serialreplagrank",
    "subject_serial_rep_lagrank",
    "tabulate_trial",
    "test_first_second_bias",
]

from collections.abc import Sequence

import numpy as np
import pandas as pd
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import apply_by_subject

from ..analyses._transitions import absolute_lag_counts, remove_positions
from ..analyses.statistics import compare_paired, mean_interval
from ..plotting import format_analysis_axes, prepare_plot_inputs
from ..repetition import equivalent_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset

# ---------------------------------------------------------------------------
# Core Tabulation
# ---------------------------------------------------------------------------


class SerialRepLagRankTabulation(Pytree):
    """Per-presentation lag-rank tabulator conditioned on serial run-up."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_lag: int = 4,
        size: int = 2,
    ) -> None:
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.list_length, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: equivalent_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.k_indices = jnp.arange(size, dtype=int)
        self.rank_sum = jnp.zeros(size, dtype=jnp.float32)
        self.transition_count = jnp.zeros(size, dtype=jnp.int32)

        self.previous_positions = lax.cond(
            first_recall > 0,
            lambda: self.item_study_positions[first_recall - 1],
            lambda: jnp.zeros((self.size,), dtype=int),
        )
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self._available_recalls_after(first_recall)

        self.has_tabulated = jnp.bool_(False)
        self.has_errored = jnp.bool_(False)

    def _available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for all study positions of a recall."""
        study_positions = self.item_study_positions[recall - 1]
        return remove_positions(self.avail_recalls, study_positions)

    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Absolute-lag bins reachable from *pos* to available items."""
        return absolute_lag_counts(
            pos, self.all_positions, self.avail_recalls, self.all_positions.size - 1
        )

    def actual_lags_from_to(
        self, prev_pos: Int_, recall_positions: Integer[Array, " size"]
    ) -> Bool[Array, " lags"]:
        """Absolute-lag bins from *prev_pos* to *recall_positions*."""

        def lags_to_recalled_position(
            recall_position: Integer[Array, ""],
        ) -> Integer[Array, " lags"]:
            return lax.cond(
                (prev_pos * recall_position) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(recall_position - prev_pos)].add(1),
            )

        return lax.map(lags_to_recalled_position, recall_positions).sum(0).astype(bool)

    def rank_at_k(
        self, k: Int_, recall_positions: Integer[Array, " size"]
    ) -> tuple[Float, Bool]:
        """Rank the transition for the k-th presentation of the previous item."""
        prev_pos = self.previous_positions[k]

        actual = self.actual_lags_from_to(prev_pos, recall_positions)
        avail = self.available_lags_from(prev_pos).astype(bool)

        cumulative = jnp.cumsum(avail)
        total = cumulative[-1]
        n_smaller = cumulative - avail

        has_choice = (total > 1) & (prev_pos > 0)
        denom = jnp.maximum(total - 1, 1).astype(jnp.float32)
        ranks = (total - 1 - n_smaller).astype(jnp.float32) / denom

        chosen = actual.astype(jnp.float32)
        mean_rank = jnp.sum(ranks * chosen) / jnp.maximum(jnp.sum(chosen), 1)

        rank = jnp.where(has_choice, mean_rank, jnp.float32(0.0))
        return rank, has_choice

    def should_tabulate(self) -> Bool[Array, ""]:
        """Return True when previous item qualifies and serial run-up holds."""
        spacing = (self.previous_positions.size > 1) & (
            (self.previous_positions[-1] - self.previous_positions[-2]) > self.min_lag
        )
        return spacing & (~self.has_tabulated) & (~self.has_errored)

    def _tabulate(self, recall: Int_) -> "SerialRepLagRankTabulation":
        """Rank the transition per presentation and update state."""
        recall_positions = self.item_study_positions[recall - 1]

        ranks, valids = lax.map(
            lambda k: self.rank_at_k(k, recall_positions),
            self.k_indices,
        )

        return self.replace(
            rank_sum=self.rank_sum + ranks,
            transition_count=self.transition_count + valids.astype(jnp.int32),
            previous_positions=self.item_study_positions[recall - 1],
            avail_recalls=self._available_recalls_after(recall),
            has_tabulated=jnp.bool_(True),
        )

    def conditional_tabulate(
        self, recall: Int_, recall_idx: Int_
    ) -> "SerialRepLagRankTabulation":
        """Rank only when previous item qualifies and serial order holds."""
        recall_study_positions = self.item_study_positions[recall - 1]
        has_errored = jnp.logical_or(
            ~jnp.any(recall_study_positions == recall_idx + 1),
            self.has_errored,
        )

        return lax.cond(
            self.should_tabulate(),
            lambda: self._tabulate(recall).replace(has_errored=has_errored),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self._available_recalls_after(recall),
                has_errored=has_errored,
            ),
        )

    def tabulate(self, recall: Int_, recall_idx: Int_) -> "SerialRepLagRankTabulation":
        """Tabulate lag-rank for this transition."""
        return lax.cond(
            recall > 0,
            lambda: self.conditional_tabulate(recall, recall_idx),
            lambda: self,
        )


# ---------------------------------------------------------------------------
# Trial and dataset functions
# ---------------------------------------------------------------------------


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_lag: int = 4,
    size: int = 2,
) -> tuple[Float[Array, " size"], Integer[Array, " size"]]:
    """Tabulate serial per-presentation lag-rank for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall sequence (1-indexed, 0-padded).
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    Returns
    -------
    tuple[Float[Array, " size"], Integer[Array, " size"]]
        Per-presentation rank sums and transition counts.

    """
    init = SerialRepLagRankTabulation(presentation, trial[0], min_lag, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i], i), init)
    return tab.rank_sum, tab.transition_count


def serialreplagrank(
    dataset: RecallDataset,
    min_lag: int = 4,
    size: int = 2,
) -> Float[Array, "trials size"]:
    """Compute serial per-trial per-presentation lag-rank factors.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls`` and ``pres_itemnos``.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    Returns
    -------
    Float[Array, "trials size"]
        Per-trial temporal factor for each presentation index.

    """
    rank_sums, counts = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        dataset["recalls"], dataset["pres_itemnos"], min_lag, size
    )
    return rank_sums / counts


# ---------------------------------------------------------------------------
# Subject-level aggregation
# ---------------------------------------------------------------------------


def subject_serial_rep_lagrank(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    min_lag: int = 4,
    size: int = 2,
) -> Float[np.ndarray, "subjects occurrences"]:
    """Compute per-subject per-presentation serial lag-rank factor.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset.
    trial_mask : Bool[Array, " trial_count"]
        Boolean mask selecting trials.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.

    Returns
    -------
    np.ndarray
        Shape ``(n_subjects, size)``.

    """
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(serialreplagrank, static_argnames=("min_lag", "size")),
        min_lag=min_lag,
        size=size,
    )
    return np.stack([np.nanmean(np.array(v), axis=0) for v in subject_values])


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def test_first_second_bias(
    observed: Float[np.ndarray, "subjects occurrences"],
    control: Float[np.ndarray, "subjects occurrences"],
) -> pd.DataFrame:
    """Compare first-minus-second occurrence scores between observed and control data.

    Parameters
    ----------
    observed : array
        Rank scores with axes participant and occurrence. Participant rows must align across inputs.
    control : array
        Rank scores with axes participant and occurrence. Participant rows must align across inputs.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns documented by ``statistics.compare_paired``. The single row compares the first-minus-second rank difference.

    Notes
    -----
    Construct differences within each participant before inference. Each row uses
    complete finite pairs for both its means and two-sided tests. A positive
    mean difference indicates a larger observed contrast than control. P-values
    are unadjusted.
    """
    observed_bias = observed[:, 0] - observed[:, 1]
    control_bias = control[:, 0] - control[:, 1]
    return compare_paired(observed_bias, control_bias, labels=["first-minus-second"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_serial_rep_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    min_lag: int = 4,
    size: int = 2,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot serial per-presentation lag-rank factors as points with error bars.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    min_lag : int
        Minimum spacing between repeated presentations.
    size : int
        Maximum presentations per item.
    color_cycle : list[str] or None
        Colors for each point.
    labels : Sequence[str] or None
        Labels for each presentation point.
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
        Matplotlib Axes with grouped serial lag-rank points and error bars.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [f"Pres {k + 1}" for k in range(size)]

    x = np.arange(size)
    width = 0.6 / max(len(datasets), 1)

    for d_idx, data in enumerate(datasets):
        factors = subject_serial_rep_lagrank(
            data, trial_masks[d_idx], min_lag=min_lag, size=size
        )
        offset = (d_idx - (len(datasets) - 1) / 2) * width

        for k in range(size):
            col = factors[:, k]
            valid = col[np.isfinite(col)]
            color = color_cycle[(d_idx * size + k) % len(color_cycle)]

            interval = mean_interval(
                valid,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            mean = interval["mean"]
            yerr = [[mean - interval["low"]], [interval["high"] - mean]]

            label = labels[k] if d_idx == 0 else None
            axis.errorbar(
                x[k] + offset,
                mean,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor="black",
                capsize=5,
                linestyle="none",
                markersize=6,
                label=label,
                zorder=3,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    axis.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    format_analysis_axes(axis, "", "Organization Score", contrast_name)
    return axis
