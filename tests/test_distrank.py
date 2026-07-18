"""Tests for jaxcmr.analyses.distrank — Distance-Rank Semantic Factor Score."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.distrank import (
    DistanceRankTabulation,
    DistRankComparisonResult,
    DistRankTestResult,
    distrank,
    percentile_rank,
    plot_distrank,
    subject_distrank,
    tabulate_trial,
)
from jaxcmr.analyses.distrank import test_distrank as run_test_distrank
from jaxcmr.analyses.distrank import (
    test_distrank_vs_comparison as run_test_comparison,
)
from jaxcmr.helpers import make_dataset


def _point_count(axis: Axes) -> int:
    return sum(len(line.get_xdata()) for line in axis.lines if line.get_marker() == "o")


def _horizontal_line_count(axis: Axes, y: float = 0.5) -> int:
    count = 0
    for line in axis.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size > 1 and np.allclose(ydata, y):
            count += 1
    return count


def make_dist_dataset(
    recalls,
    pres_itemids=None,
    subject=1,
):
    """Construct a dataset with Distance-CRP-style pres_itemids."""
    ds = make_dataset(recalls, pres_itemnos=pres_itemids, subject=subject)
    ds["pres_itemids"] = ds["pres_itemnos"]
    return ds


DISTANCES = jnp.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=jnp.float32,
)


# ---------------------------------------------------------------------------
# 1. percentile_rank — core utility
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_nearest(self):
        """Behavior: nearest item gets rank 1.0."""
        pool = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = percentile_rank(jnp.float32(1.0), pool)
        assert float(result) == pytest.approx(1.0)

    def test_furthest(self):
        """Behavior: furthest item gets rank 0.0."""
        pool = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = percentile_rank(jnp.float32(4.0), pool)
        assert float(result) == pytest.approx(0.0)

    def test_midpoint_tie(self):
        """Behavior: tied items get midpoint rank."""
        pool = jnp.array([1.0, 2.0, 2.0, 3.0])
        result = percentile_rank(jnp.float32(2.0), pool)
        assert float(result) == pytest.approx(0.5)

    def test_single_element(self):
        """Behavior: single-element pool returns NaN (no choice)."""
        pool = jnp.array([5.0])
        result = percentile_rank(jnp.float32(5.0), pool)
        assert jnp.isnan(result)

    def test_inf_sentinels_excluded(self):
        """Behavior: inf entries are treated as unavailable."""
        pool = jnp.array([jnp.inf, jnp.inf, 1.0, 3.0])
        result = percentile_rank(jnp.float32(1.0), pool)
        assert float(result) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. DistanceRankTabulation — sentinel and state
# ---------------------------------------------------------------------------


class TestDistanceRankTabulation:
    def test_zero_recall_is_noop(self):
        """Behavior: zero recall does not change state."""
        tab = DistanceRankTabulation(
            jnp.array([True, True, True]),
            jnp.int32(1),
            DISTANCES,
        )
        updated = tab.tabulate(jnp.int32(0))
        assert float(updated.rank_sum) == 0.0
        assert int(updated.transition_count) == 0

    def test_first_recall_unavailable(self):
        """Behavior: first recall position is marked unavailable."""
        tab = DistanceRankTabulation(
            jnp.array([True, True, True]),
            jnp.int32(2),
            DISTANCES,
        )
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(tab.avail_items, expected)

    def test_no_choice_transition_excluded(self):
        """Behavior: final transition with one candidate is not counted."""
        trial = jnp.array([1, 2, 3], dtype=jnp.int32)
        pres = jnp.array([1, 2, 3], dtype=jnp.int32)
        tab_factor = tabulate_trial(trial, pres, DISTANCES)
        assert float(tab_factor) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. tabulate_trial / distrank — known values
# ---------------------------------------------------------------------------


class TestDistrank:
    def test_perfect_nearest_transition(self):
        """Behavior: choosing the nearest available item gives factor 1.0."""
        trial = jnp.array([1, 2, 3], dtype=jnp.int32)
        pres = jnp.array([1, 2, 3], dtype=jnp.int32)
        factor = tabulate_trial(trial, pres, DISTANCES)
        assert float(factor) == pytest.approx(1.0)

    def test_furthest_transition(self):
        """Behavior: choosing the farthest available item gives factor 0.0."""
        trial = jnp.array([1, 3, 2], dtype=jnp.int32)
        pres = jnp.array([1, 2, 3], dtype=jnp.int32)
        factor = tabulate_trial(trial, pres, DISTANCES)
        assert float(factor) == pytest.approx(0.0)

    def test_midpoint_tie_transition(self):
        """Behavior: equal nearest distances get midpoint rank."""
        distances = jnp.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        trial = jnp.array([1, 2, 3], dtype=jnp.int32)
        pres = jnp.array([1, 2, 3], dtype=jnp.int32)
        factor = tabulate_trial(trial, pres, distances)
        assert float(factor) == pytest.approx(0.5)

    def test_distrank_returns_per_trial(self):
        """Behavior: distrank returns one factor per trial."""
        ds = make_dist_dataset(
            recalls=jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32),
            pres_itemids=jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        )
        factors = distrank(ds, DISTANCES)
        np.testing.assert_allclose(np.array(factors), np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# 4. JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    def test_distrank_jit_compatible(self):
        """Behavior: JIT compilation produces same results."""
        ds = make_dist_dataset(
            recalls=jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32),
            pres_itemids=jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        )
        eager = distrank(ds, DISTANCES)
        compiled = jit(distrank)(ds, DISTANCES)
        np.testing.assert_allclose(np.array(eager), np.array(compiled), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Statistical tests
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_returns_dataclass(self):
        """Behavior: test_distrank returns DistRankTestResult."""
        factors = np.array(
            [0.7, 0.8, 0.6, 0.75, 0.65, 0.72, 0.68, 0.73, 0.69, 0.71, 0.66, 0.74]
        )
        result = run_test_distrank(factors)
        assert isinstance(result, DistRankTestResult)
        assert result.n == 12
        assert result.mean_factor > 0.5
        assert result.t_pval < 0.05

    def test_chance_factors(self):
        """Behavior: factors near chance yield non-significant test."""
        rng = np.random.RandomState(42)
        factors = 0.5 + rng.normal(0, 0.01, size=20)
        result = run_test_distrank(factors)
        assert result.t_pval > 0.05

    def test_vs_comparison_identical(self):
        """Behavior: identical arrays yield non-significant comparison."""
        a = np.array(
            [0.6, 0.7, 0.55, 0.65, 0.72, 0.68, 0.61, 0.73, 0.59, 0.67, 0.64, 0.71]
        )
        result = run_test_comparison(a, a.copy())
        assert isinstance(result, DistRankComparisonResult)
        assert abs(result.mean_diff) < 1e-10


# ---------------------------------------------------------------------------
# 6. subject_distrank
# ---------------------------------------------------------------------------


class TestSubjectDistrank:
    def test_returns_per_subject(self):
        """Behavior: returns one scalar per subject."""
        ds = make_dist_dataset(
            recalls=jnp.array(
                [[1, 2, 3], [1, 3, 2], [1, 2, 3], [1, 3, 2]],
                dtype=jnp.int32,
            ),
            pres_itemids=jnp.array(
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                dtype=jnp.int32,
            ),
            subject=jnp.array([[1], [1], [2], [2]]),
        )
        mask = jnp.ones(4, dtype=bool)
        factors = subject_distrank(ds, mask, DISTANCES)
        assert len(factors) == 2
        assert all(np.isfinite(factors))


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------


class TestPlotDistrank:
    def test_returns_axes_with_distance_matrix(self):
        """Behavior: plot_distrank returns an Axes object."""
        ds = make_dist_dataset(
            recalls=jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32),
            pres_itemids=jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        )
        mask = jnp.ones(2, dtype=bool)
        ax = plot_distrank(ds, mask, distance_matrix=DISTANCES)
        assert isinstance(ax, Axes)
        assert _point_count(ax) == 1
        assert len(ax.patches) == 0
        assert _horizontal_line_count(ax) == 0
        np.testing.assert_allclose(ax.get_xlim(), (-0.5, 0.5))
        assert ax.get_ylabel() == "Organization Score"
        plt.close("all")

    def test_returns_axes_with_features(self):
        """Behavior: plot_distrank can compute cosine distances from features."""
        ds = make_dist_dataset(
            recalls=jnp.array([[1, 2, 3], [1, 3, 2]], dtype=jnp.int32),
            pres_itemids=jnp.array([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32),
        )
        features = jnp.eye(3, dtype=jnp.float32)
        mask = jnp.ones(2, dtype=bool)
        ax = plot_distrank(ds, mask, features=features)
        assert isinstance(ax, Axes)
        assert _point_count(ax) == 1
        assert len(ax.patches) == 0
        assert _horizontal_line_count(ax) == 0
        np.testing.assert_allclose(ax.get_xlim(), (-0.5, 0.5))
        assert ax.get_ylabel() == "Organization Score"
        plt.close("all")

    def test_rejects_neither_input_type(self):
        """Behavior: plot_distrank requires one input type."""
        ds = make_dist_dataset(jnp.array([[1, 2, 3]], dtype=jnp.int32))
        mask = jnp.ones(1, dtype=bool)
        with pytest.raises(ValueError):
            plot_distrank(ds, mask)

    def test_rejects_both_input_types(self):
        """Behavior: plot_distrank rejects ambiguous input types."""
        ds = make_dist_dataset(jnp.array([[1, 2, 3]], dtype=jnp.int32))
        mask = jnp.ones(1, dtype=bool)
        with pytest.raises(ValueError):
            plot_distrank(
                ds,
                mask,
                features=jnp.eye(3, dtype=jnp.float32),
                distance_matrix=DISTANCES,
            )
