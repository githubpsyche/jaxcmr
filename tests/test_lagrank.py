"""Tests for jaxcmr.analyses.lagrank — Lag-Rank Temporal Factor Score."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp

from jaxcmr.analyses.lagrank import (
    LagRankComparisonResult,
    LagRankTestResult,
    SimpleTabulation,
    Tabulation,
    lagrank,
    percentile_rank,
    plot_lagrank,
    simple_lagrank,
    simple_tabulate_trial,
    subject_lagrank,
    tabulate_trial,
)
from jaxcmr.analyses.lagrank import test_lagrank as run_test_lagrank
from jaxcmr.analyses.lagrank import test_lagrank_vs_comparison as run_test_comparison
from jaxcmr.typing import RecallDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(recalls, presentations) -> RecallDataset:
    """Wrap arrays into a RecallDataset dict."""
    recalls = jnp.asarray(recalls, dtype=jnp.int32)
    presentations = jnp.asarray(presentations, dtype=jnp.int32)
    n_trials, _ = recalls.shape
    list_length = presentations.shape[1]
    return {
        "subject": jnp.ones((n_trials, 1), dtype=jnp.int32),
        "listLength": jnp.full((n_trials, 1), list_length, dtype=jnp.int32),
        "pres_itemnos": presentations,
        "recalls": recalls,
    }


def _simple_dataset(recalls, list_length) -> RecallDataset:
    """Shorthand for unique-item lists: presentations = [[1..L], ...]."""
    recalls = jnp.asarray(recalls, dtype=jnp.int32)
    n_trials = recalls.shape[0]
    pres = jnp.tile(jnp.arange(1, list_length + 1, dtype=jnp.int32), (n_trials, 1))
    return _make_dataset(recalls, pres)


# ---------------------------------------------------------------------------
# 1. percentile_rank — core utility
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_nearest(self):
        """Behavior: nearest item gets rank 1.0.
        Given: pool = [1, 2, 3, 4], target = 1
        When: percentile_rank is computed
        Then: returns 1.0
        Why this matters: fundamental property of the ranking.
        """
        pool = jnp.array([1, 2, 3, 4])
        result = percentile_rank(jnp.int32(1), pool)
        assert float(result) == pytest.approx(1.0)

    def test_furthest(self):
        """Behavior: furthest item gets rank 0.0.
        Given: pool = [1, 2, 3, 4], target = 4
        When: percentile_rank is computed
        Then: returns 0.0
        Why this matters: fundamental property of the ranking.
        """
        pool = jnp.array([1, 2, 3, 4])
        result = percentile_rank(jnp.int32(4), pool)
        assert float(result) == pytest.approx(0.0)

    def test_midpoint_tie(self):
        """Behavior: tied items get midpoint rank.
        Given: pool = [1, 2, 2, 3], target = 2
        When: percentile_rank is computed
        Then: returns 0.5 (midpoint of tied positions)
        Why this matters: tie-breaking must use midpoint convention.
        """
        pool = jnp.array([1, 2, 2, 3])
        result = percentile_rank(jnp.int32(2), pool)
        assert float(result) == pytest.approx(0.5)

    def test_single_element(self):
        """Behavior: single-element pool returns NaN (no choice).
        Given: pool = [5], target = 5
        When: percentile_rank is computed
        Then: returns NaN without error
        Why this matters: edge case with no meaningful rank.
        """
        pool = jnp.array([5])
        result = percentile_rank(jnp.int32(5), pool)
        assert jnp.isnan(result)

    def test_negative_sentinels_excluded(self):
        """Behavior: negative entries are treated as missing.
        Given: pool = [-1, -1, 1, 3], target = 1
        When: percentile_rank is computed
        Then: ranks among valid entries only (1 and 3), returns 1.0
        Why this matters: sentinel filtering must work correctly.
        """
        pool = jnp.array([-1, -1, 1, 3])
        result = percentile_rank(jnp.int32(1), pool)
        assert float(result) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. SimpleTabulation — sentinel and state
# ---------------------------------------------------------------------------


class TestSimpleTabulation:
    def test_zero_recall_is_noop(self):
        """Behavior: zero recall does not change state.
        Given: SimpleTabulation(L=4, first=2)
        When: update(0) is called
        Then: rank_sum and transition_count are unchanged
        Why this matters: zeros are padding sentinels.
        """
        tab = SimpleTabulation(4, jnp.int32(2))
        updated = tab.update(jnp.int32(0))
        assert float(updated.rank_sum) == 0.0
        assert int(updated.transition_count) == 0

    def test_first_recall_unavailable(self):
        """Behavior: first recall position is marked unavailable.
        Given: SimpleTabulation(L=4, first=2)
        When: inspecting avail_items
        Then: position 2 is False, others are True
        Why this matters: availability tracking is foundational.
        """
        tab = SimpleTabulation(4, jnp.int32(2))
        expected = jnp.array([True, False, True, True])
        assert jnp.array_equal(tab.avail_items, expected)

    def test_transition_count(self):
        """Behavior: transition count increments correctly.
        Given: L=4, recalls [2, 3, 1] (first recall is 2)
        When: scanning through remaining recalls [3, 1]
        Then: transition_count == 2
        Why this matters: count must match number of valid transitions.
        """
        trial = jnp.array([2, 3, 1, 0], dtype=jnp.int32)
        tab = simple_tabulate_trial(trial, 4)
        assert int(tab.transition_count) == 2


# ---------------------------------------------------------------------------
# 3. simple_tabulate_trial / simple_lagrank — known values
# ---------------------------------------------------------------------------


class TestSimpleLagrank:
    def test_perfect_contiguity(self):
        """Behavior: sequential recall gives factor 1.0.
        Given: L=4, trial=[1,2,3,4] (always nearest neighbor)
        When: simple_tabulate_trial is called
        Then: rank_sum / transition_count == 1.0
        Why this matters: perfect contiguity is the upper bound.
        """
        trial = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        tab = simple_tabulate_trial(trial, 4)
        factor = float(tab.rank_sum / tab.transition_count)
        assert factor == pytest.approx(1.0)

    def test_single_transition(self):
        """Behavior: single transition gets correct rank.
        Given: L=5, trial=[3,5,0,0,0]
        When: simple_lagrank computes the factor
        Then: factor matches hand-computed bin-based rank

        Hand computation (bin-based):
          previous=3, available=[1,2,4,5]
          available absolute lag bins: {1, 2} (2 unique bins)
          actual |lag| = |5-3| = 2 (the larger bin)
          rank = (2-1-1) / max(2-1, 1) = 0.0

        Why this matters: verifies the bin-based ranking formula.
        """
        trial = jnp.array([[3, 5, 0, 0, 0]], dtype=jnp.int32)
        factors = simple_lagrank(trial, 5)
        assert float(factors[0]) == pytest.approx(0.0, abs=1e-4)

    def test_reverse_order(self):
        """Behavior: reverse sequential recall gives factor 1.0.
        Given: L=4, trial=[4,3,2,1]
        When: simple_lagrank computes the factor
        Then: factor == 1.0 (always nearest neighbor, just backwards)
        Why this matters: direction doesn't matter for absolute lag rank.
        """
        trial = jnp.array([[4, 3, 2, 1]], dtype=jnp.int32)
        factors = simple_lagrank(trial, 4)
        assert float(factors[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Tabulation — repeated-item handling
# ---------------------------------------------------------------------------


class TestTabulation:
    def test_repeated_recall_ignored(self):
        """Behavior: recalling an already-recalled item is skipped.
        Given: pres=[1,2,3,4], recalls=[3,3,4]
        When: Tabulation processes the sequence
        Then: transition_count == 1 (only 3→4 counted, not 3→3)
        Why this matters: re-recalls have no valid study positions.
        """
        pres = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        tab = Tabulation(pres, jnp.int32(3), size=1)
        tab = tab.tabulate(jnp.int32(3))  # repeat — should be skipped
        tab = tab.tabulate(jnp.int32(4))  # valid transition
        assert int(tab.transition_count) == 1

    def test_zero_padding_safe(self):
        """Behavior: zero-padded recalls don't crash or corrupt state.
        Given: pres=[10,20,30,40], size=3, recalls=[1,2,0,0]
        When: Tabulation processes the sequence
        Then: no crash; transition_count == 1
        Why this matters: zero-padding is ubiquitous in recall arrays.
        """
        pres = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
        tab = Tabulation(pres, jnp.int32(1), size=3)
        tab = tab.tabulate(jnp.int32(2))
        tab = tab.tabulate(jnp.int32(0))
        tab = tab.tabulate(jnp.int32(0))
        assert int(tab.transition_count) == 1

    def test_repeated_item_positions(self):
        """Behavior: repeated items track all study positions.
        Given: pres=[1,2,1,3] (item 1 at positions 1 and 3), size=2
        When: inspecting item_study_positions
        Then: item 1's positions are [1,3], item 2's are [2,0]
        Why this matters: correct position tracking for repeats.
        """
        pres = jnp.array([1, 2, 1, 3], dtype=jnp.int32)
        tab = Tabulation(pres, jnp.int32(1), size=2)
        # item_study_positions[0] = positions for item at position 1 (item 1)
        pos_item1 = tab.item_study_positions[0]
        assert list(pos_item1) == [1, 3]
        # item_study_positions[1] = positions for item at position 2 (item 2)
        pos_item2 = tab.item_study_positions[1]
        assert list(pos_item2) == [2, 0]


# ---------------------------------------------------------------------------
# 5. tabulate_trial / lagrank — integration
# ---------------------------------------------------------------------------


class TestLagrank:
    def test_tabulate_trial_returns_scalar(self):
        """Behavior: tabulate_trial returns a scalar float.
        Given: pres=[1,2,3], trial=[1,2,3]
        When: tabulate_trial is called
        Then: result is a scalar in (0, 1]
        Why this matters: downstream code expects a scalar per trial.
        """
        pres = jnp.array([1, 2, 3], dtype=jnp.int32)
        trial = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = tabulate_trial(trial, pres, size=1)
        assert result.ndim == 0
        assert 0.0 < float(result) <= 1.0

    def test_lagrank_dataset_interface(self):
        """Behavior: lagrank returns 1-D per-trial factors.
        Given: dataset with 3 unique-item trials
        When: lagrank is called
        Then: returns 1-D array of length 3
        Why this matters: confirms the dataset API works.
        """
        recalls = [[1, 2, 3, 0], [3, 2, 1, 0], [1, 3, 2, 0]]
        pres = [[1, 2, 3, 4]] * 3
        ds = _make_dataset(recalls, pres)
        factors = lagrank(ds, size=1)
        assert factors.shape == (3,)
        assert all(jnp.isfinite(factors))


# ---------------------------------------------------------------------------
# 6. Simple vs Full agreement
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_simple_and_full_agree_unique_items(self):
        """Behavior: both tabulators agree for unique-item lists.
        Given: unique-item dataset (size=1)
        When: simple_lagrank and lagrank both compute factors
        Then: per-trial factors are allclose
        Why this matters: SimpleTabulation is an optimization of Tabulation.
        """
        recalls = [
            [1, 2, 3, 4, 0],
            [3, 1, 4, 2, 0],
            [5, 4, 3, 2, 1],
        ]
        list_length = 5
        ds = _simple_dataset(recalls, list_length)
        trials = jnp.asarray(recalls, dtype=jnp.int32)

        simple_factors = simple_lagrank(trials, list_length)
        full_factors = lagrank(ds, size=1)
        np.testing.assert_allclose(
            np.array(simple_factors), np.array(full_factors), atol=1e-5
        )


# ---------------------------------------------------------------------------
# 7. JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    def test_lagrank_jit_compatible(self):
        """Behavior: JIT compilation produces same results.
        Given: a dataset
        When: lagrank is called with and without JIT
        Then: results are allclose
        Why this matters: JIT is the expected usage pattern.
        """
        ds = _simple_dataset([[1, 2, 3, 0], [3, 2, 1, 0]], 4)
        eager = lagrank(ds, size=1)
        compiled = jit(lagrank, static_argnames=("size",))(ds, size=1)
        np.testing.assert_allclose(
            np.array(eager), np.array(compiled), atol=1e-6
        )


# ---------------------------------------------------------------------------
# 8. Statistical tests
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_returns_dataclass(self):
        """Behavior: test_lagrank returns LagRankTestResult.
        Given: array of factors > 0.5
        When: test_lagrank is called
        Then: returns LagRankTestResult with correct fields
        Why this matters: downstream code depends on the result structure.
        """
        factors = np.array([0.7, 0.8, 0.6, 0.75, 0.65, 0.72, 0.68,
                            0.73, 0.69, 0.71, 0.66, 0.74])
        result = run_test_lagrank(factors)
        assert isinstance(result, LagRankTestResult)
        assert result.n == 12
        assert result.mean_factor > 0.5
        assert result.t_pval < 0.05

    def test_chance_factors(self):
        """Behavior: factors near chance yield non-significant test.
        Given: array of factors jittered around 0.5
        When: test_lagrank is called
        Then: t_pval is not significant (cannot reject H0)
        Why this matters: the test should not reject the null when true.
        """
        rng = np.random.RandomState(42)
        factors = 0.5 + rng.normal(0, 0.01, size=20)
        result = run_test_lagrank(factors)
        assert result.t_pval > 0.05

    def test_vs_comparison_identical(self):
        """Behavior: identical arrays yield non-significant comparison.
        Given: two identical factor arrays
        When: test_lagrank_vs_comparison is called
        Then: mean_diff ≈ 0, t_pval is not significant
        Why this matters: paired test should not reject for no difference.
        """
        a = np.array([0.6, 0.7, 0.55, 0.65, 0.72, 0.68,
                       0.61, 0.73, 0.59, 0.67, 0.64, 0.71])
        result = run_test_comparison(a, a.copy())
        assert isinstance(result, LagRankComparisonResult)
        assert abs(result.mean_diff) < 1e-10

    def test_str_representation(self):
        """Behavior: __str__ returns a non-empty string.
        Given: a LagRankTestResult instance
        When: str() is called
        Then: returns a multi-line summary
        Why this matters: result must be printable.
        """
        result = LagRankTestResult(
            n=10, mean_factor=0.7, t_stat=3.5, t_pval=0.005,
            w_stat=45.0, w_pval=0.01,
        )
        s = str(result)
        assert len(s) > 0
        assert "0.7" in s


# ---------------------------------------------------------------------------
# 9. subject_lagrank
# ---------------------------------------------------------------------------


class TestSubjectLagrank:
    def test_returns_per_subject(self):
        """Behavior: returns one scalar per subject.
        Given: dataset with 2 subjects
        When: subject_lagrank is called
        Then: returns array of length 2
        Why this matters: subject-level aggregation is needed for stats.
        """
        recalls = jnp.array(
            [[1, 2, 3, 0], [3, 2, 1, 0], [1, 3, 2, 0], [2, 1, 3, 0]],
            dtype=jnp.int32,
        )
        pres = jnp.tile(jnp.arange(1, 4, dtype=jnp.int32), (4, 1))
        # pad pres to match list_length if needed
        pres = jnp.concatenate(
            [pres, jnp.full((4, 1), 4, dtype=jnp.int32)], axis=1
        )
        subjects = jnp.array([[1], [1], [2], [2]], dtype=jnp.int32)
        ds: RecallDataset = {
            "subject": subjects,
            "listLength": jnp.full((4, 1), 4, dtype=jnp.int32),
            "pres_itemnos": pres,
            "recalls": recalls,
        }
        mask = jnp.ones(4, dtype=bool)
        factors = subject_lagrank(ds, mask, size=1)
        assert len(factors) == 2
        assert all(np.isfinite(factors))


# ---------------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------------


class TestPlotLagrank:
    def test_returns_axes(self):
        """Behavior: plot_lagrank returns an Axes object.
        Given: simple dataset and mask
        When: plot_lagrank is called
        Then: returns matplotlib Axes
        Why this matters: plotting API must be consistent.
        """
        ds = _simple_dataset([[1, 2, 3, 0], [3, 2, 1, 0]], 4)
        mask = jnp.ones(2, dtype=bool)
        ax = plot_lagrank(ds, mask, size=1)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_multiple_conditions(self):
        """Behavior: multiple datasets can be plotted.
        Given: two datasets and two masks
        When: plot_lagrank is called
        Then: returns Axes without error
        Why this matters: multi-condition comparison is common.
        """
        ds1 = _simple_dataset([[1, 2, 3, 0], [3, 2, 1, 0]], 4)
        ds2 = _simple_dataset([[2, 1, 3, 0], [1, 3, 2, 0]], 4)
        masks = [jnp.ones(2, dtype=bool), jnp.ones(2, dtype=bool)]
        ax = plot_lagrank([ds1, ds2], masks, labels=["A", "B"], size=1)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")
