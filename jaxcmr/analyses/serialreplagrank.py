"""Serial repetition lag-rank temporal factor score.

Computes per-presentation temporal factor scores for transitions FROM
repeated items, restricted to the first qualifying transition during a
correct serial run-up. Parallels ``serialrepcrp`` but produces a scalar
factor per presentation index instead of a CRP curve.

Notes
-----
- Only the FIRST qualifying transition per trial is tabulated. Once a
  transition is ranked (``has_tabulated``) or the serial run-up is broken
  (``has_errored``), no further ranking occurs on that trial.
- A serial run-up means items are recalled in the order they were studied
  (study position 1 first, study position 2 second, etc.). If any recall
  breaks this order, ``has_errored`` is set.
- When a qualifying transition is found, ranking follows the same
  cumulative-bin approach as ``replagrank``: absolute lags from each
  study position of the previous item are ranked among available lags.

"""

__all__ = [
    "SerialRepLagRankTabulation",
    "tabulate_trial",
    "serialreplagrank",
    "subject_serial_rep_lagrank",
    "SerialRepLagRankTestResult",
    "test_serial_rep_lagrank_vs_control",
    "test_first_second_bias",
    "plot_serial_rep_lagrank",
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
from ..plotting import prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from .crp import set_false_at_index


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
    ):
        self.size = size
        self.min_lag = min_lag
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.list_length, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
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

    def actual_lags_from_to(
        self, prev_pos: Int_, recall_positions: Integer[Array, " size"]
    ) -> Bool[Array, " lags"]:
        """Absolute-lag bins from *prev_pos* to *recall_positions*."""

        def f(rp):
            return lax.cond(
                (prev_pos * rp) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[jnp.abs(rp - prev_pos)].add(1),
            )

        return lax.map(f, recall_positions).sum(0).astype(bool)

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

    def should_tabulate(self) -> Bool:
        """Return True when previous item qualifies and serial run-up holds."""
        spacing = (
            (self.previous_positions.size > 1)
            & ((self.previous_positions[-1] - self.previous_positions[-2]) > self.min_lag)
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

    def tabulate(
        self, recall: Int_, recall_idx: Int_
    ) -> "SerialRepLagRankTabulation":
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
    tab = lax.fori_loop(
        1, trial.size, lambda i, t: t.tabulate(trial[i], i), init
    )
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
) -> np.ndarray:
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


@dataclass
class SerialRepLagRankTestResult:
    """Results from a serial per-presentation lag-rank test."""

    presentations: np.ndarray
    mean_observed: np.ndarray
    mean_comparison: np.ndarray
    mean_diffs: np.ndarray
    t_stats: np.ndarray
    t_pvals: np.ndarray
    w_stats: np.ndarray
    w_pvals: np.ndarray

    def __str__(self) -> str:
        lines = [
            f"{'Pres':>5} | {'Mean Obs':>9} {'Mean Ctrl':>10} {'Diff':>8} | "
            f"{'t-stat':>8} {'t p-val':>10} | {'W-stat':>8} {'W p-val':>10}",
            f"{'-'*5}-+-{'-'*30}-+-{'-'*20}-+-{'-'*20}",
        ]
        for i, pres in enumerate(self.presentations):
            lines.append(
                f"{pres:>5} | {self.mean_observed[i]:>9.4f} {self.mean_comparison[i]:>10.4f} "
                f"{self.mean_diffs[i]:>8.4f} | "
                f"{self.t_stats[i]:>8.3f} {self.t_pvals[i]:>10.4f} | "
                f"{self.w_stats[i]:>8.1f} {self.w_pvals[i]:>10.4f}"
            )
        return "\n".join(lines)


def test_serial_rep_lagrank_vs_control(
    observed: np.ndarray,
    control: np.ndarray,
) -> dict[str, SerialRepLagRankTestResult]:
    """Test observed vs control serial lag-rank factors.

    Parameters
    ----------
    observed : np.ndarray
        Per-subject factors, shape ``(n_subjects, size)``.
    control : np.ndarray
        Per-subject control factors, same shape.

    Returns
    -------
    dict[str, SerialRepLagRankTestResult]
        Results keyed by presentation label.

    """
    size = observed.shape[1]
    presentations = np.arange(1, size + 1)
    results: dict[str, SerialRepLagRankTestResult] = {}

    for rep_idx in range(size):
        obs = observed[:, rep_idx]
        ctrl = control[:, rep_idx]
        diff = obs - ctrl

        t_stat, t_pval = stats.ttest_rel(obs, ctrl, nan_policy="omit")

        valid = ~(np.isnan(obs) | np.isnan(ctrl))
        if valid.sum() > 10:
            w_stat, w_pval = stats.wilcoxon(diff[valid], alternative="two-sided")
        else:
            w_stat, w_pval = np.nan, np.nan

        label = "First Presentation" if rep_idx == 0 else f"Presentation {rep_idx + 1}"
        if rep_idx == 1:
            label = "Second Presentation"

        results[label] = SerialRepLagRankTestResult(
            presentations=presentations[rep_idx : rep_idx + 1],
            mean_observed=np.array([np.nanmean(obs)]),
            mean_comparison=np.array([np.nanmean(ctrl)]),
            mean_diffs=np.array([np.nanmean(diff)]),
            t_stats=np.array([float(t_stat)]),
            t_pvals=np.array([float(t_pval)]),
            w_stats=np.array([float(w_stat) if np.isfinite(w_stat) else np.nan]),
            w_pvals=np.array([float(w_pval) if np.isfinite(w_pval) else np.nan]),
        )

    return results


def test_first_second_bias(
    observed: np.ndarray,
    control: np.ndarray,
) -> SerialRepLagRankTestResult:
    """Test whether first-presentation serial factor bias differs from control.

    Parameters
    ----------
    observed : np.ndarray
        Per-subject factors, shape ``(n_subjects, size)``.
    control : np.ndarray
        Per-subject control factors, same shape.

    Returns
    -------
    SerialRepLagRankTestResult
        Test of (obs_first - obs_second) vs (ctrl_first - ctrl_second).

    """
    obs_diff = observed[:, 0] - observed[:, 1]
    ctrl_diff = control[:, 0] - control[:, 1]
    effect = obs_diff - ctrl_diff

    t_stat, t_pval = stats.ttest_rel(obs_diff, ctrl_diff, nan_policy="omit")

    valid = ~(np.isnan(obs_diff) | np.isnan(ctrl_diff))
    if valid.sum() > 10:
        w_stat, w_pval = stats.wilcoxon(effect[valid], alternative="two-sided")
    else:
        w_stat, w_pval = np.nan, np.nan

    return SerialRepLagRankTestResult(
        presentations=np.array([1]),
        mean_observed=np.array([np.nanmean(obs_diff)]),
        mean_comparison=np.array([np.nanmean(ctrl_diff)]),
        mean_diffs=np.array([np.nanmean(effect)]),
        t_stats=np.array([float(t_stat)]),
        t_pvals=np.array([float(t_pval)]),
        w_stats=np.array([float(w_stat) if np.isfinite(w_stat) else np.nan]),
        w_pvals=np.array([float(w_pval) if np.isfinite(w_pval) else np.nan]),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_serial_rep_lagrank(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    min_lag: int = 4,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot serial per-presentation lag-rank factors as a grouped bar chart.

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
        Colors for each bar.
    labels : Sequence[str] or None
        Labels for each presentation bar.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the grouped bar chart.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [f"Pres {k+1}" for k in range(size)]

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
            mean = float(np.nanmean(col))
            color = color_cycle[(d_idx * size + k) % len(color_cycle)]

            if len(valid) > 1:
                ci = bootstrap(
                    (valid,), np.nanmean, confidence_level=confidence_level
                ).confidence_interval
                yerr = [[mean - ci.low], [ci.high - mean]]
            else:
                yerr = [[0], [0]]

            label = labels[k] if d_idx == 0 else None
            axis.bar(
                x[k] + offset, mean, width,
                color=color, label=label, alpha=0.7,
            )
            axis.errorbar(
                x[k] + offset, mean, yerr=yerr,
                fmt="none", color="black", capsize=5,
            )

    axis.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=14)
    set_plot_labels(axis, "", "Temporal Factor", contrast_name)
    return axis
