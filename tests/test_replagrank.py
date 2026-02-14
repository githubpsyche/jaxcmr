"""Tests for jaxcmr.analyses.replagrank."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp

from jaxcmr.analyses.replagrank import (
    RepLagRankTabulation,
    replagrank,
    subject_rep_lagrank,
    plot_rep_lagrank,
    test_rep_lagrank_vs_control as run_test_vs_control,
    test_first_second_bias as run_test_bias,
    RepLagRankTestResult,
)
from jaxcmr.helpers import make_dataset


# ---- Tabulation: non-repeated items are skipped ----

class TestTabulationSkipsNonRepeated:
    """Given a previous item with only one study position."""

    def test_non_repeated_item_skipped(self):
        """When the previous item is not repeated, transition_count stays zero."""
        # Item 1 at pos 1 only, item 2 at pos 2 only, etc. — no repeats.
        pres = jnp.array([1, 2, 3, 4, 5, 6])
        tab = RepLagRankTabulation(pres, jnp.int32(1), min_lag=2, size=2)
        tab = tab.tabulate(jnp.int32(2))
        tab = tab.tabulate(jnp.int32(3))
        assert int(tab.transition_count.sum()) == 0


# ---- Tabulation: repeated items are tabulated ----

class TestTabulationRepeatedItem:
    """Given a previous item with two study positions."""

    def test_repeated_item_tabulated(self):
        """When the previous item is repeated with spacing > min_lag, transition is counted."""
        # Item 1 at positions 1 and 6 (spacing=5 > min_lag=2)
        pres = jnp.array([1, 2, 3, 4, 5, 1])
        tab = RepLagRankTabulation(pres, jnp.int32(1), min_lag=2, size=2)
        # Now previous is item 1 (positions [1,6]). Recall item 2.
        tab = tab.tabulate(jnp.int32(2))
        assert int(tab.transition_count.sum()) > 0


# ---- Tabulation: zero recall is noop ----

class TestZeroRecallNoop:
    """Given a RepLagRankTabulation and a zero recall."""

    def test_zero_recall_noop(self):
        """When recall is 0, state is unchanged."""
        pres = jnp.array([1, 2, 3, 4])
        tab = RepLagRankTabulation(pres, jnp.int32(1), min_lag=1, size=2)
        rank_before = float(tab.rank_sum.sum())
        count_before = int(tab.transition_count.sum())
        tab = tab.tabulate(jnp.int32(0))
        assert float(tab.rank_sum.sum()) == rank_before
        assert int(tab.transition_count.sum()) == count_before


# ---- Integration: shape and range ----

class TestReplagrankIntegration:
    """Given a dataset with repeated items."""

    @pytest.fixture
    def dataset(self):
        # Item 1 appears at pos 1 and 6, item 2 at pos 2 and 7
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 3, 2, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
            [3, 1, 2, 4, 5, 0, 0, 0],
        ])
        return make_dataset(recalls, pres)

    def test_replagrank_returns_shape(self, dataset):
        """replagrank returns (n_trials, size)."""
        result = replagrank(dataset, min_lag=2, size=2)
        assert result.shape == (3, 2)

    def test_replagrank_values_in_range(self, dataset):
        """All finite values are in [0, 1]."""
        result = replagrank(dataset, min_lag=2, size=2)
        finite = result[jnp.isfinite(result)]
        if finite.size > 0:
            assert float(finite.min()) >= 0.0
            assert float(finite.max()) <= 1.0


# ---- Subject aggregation ----

class TestSubjectRepLagrank:
    """Given a dataset with 2 subjects."""

    def test_subject_rep_lagrank_shape(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 3, 2, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
            [3, 1, 2, 4, 5, 0, 0, 0],
            [1, 3, 5, 2, 0, 0, 0, 0],
        ])
        subjects = jnp.array([0, 0, 1, 1])
        dataset = make_dataset(recalls, pres, subject=subjects)
        mask = jnp.ones(4, dtype=bool)
        result = subject_rep_lagrank(dataset, mask, min_lag=2, size=2)
        assert result.shape == (2, 2)


# ---- Statistical tests ----

class TestRepLagrankStats:
    """Given per-subject factor arrays."""

    def test_vs_control_returns_dict(self):
        rng = np.random.default_rng(42)
        observed = rng.uniform(0.4, 0.8, size=(20, 2))
        control = rng.uniform(0.3, 0.7, size=(20, 2))
        result = run_test_vs_control(observed, control)
        assert isinstance(result, dict)
        assert "First Presentation" in result
        assert "Second Presentation" in result
        assert isinstance(result["First Presentation"], RepLagRankTestResult)

    def test_first_second_bias_identical(self):
        rng = np.random.default_rng(42)
        factors = rng.uniform(0.4, 0.8, size=(20, 2))
        result = run_test_bias(factors, factors)
        assert result.t_pvals[0] > 0.5 or np.isnan(result.t_pvals[0])


# ---- Plotting ----

class TestPlotRepLagrank:
    """Given a dataset and mask."""

    def test_plot_returns_axes(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 3, 2, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
        ])
        dataset = make_dataset(recalls, pres)
        mask = jnp.ones(2, dtype=bool)
        ax = plot_rep_lagrank(dataset, mask, min_lag=2, size=2, labels=["1st", "2nd"])
        from matplotlib.axes import Axes
        assert isinstance(ax, Axes)


# ---- JIT compatibility ----

class TestReplagrankJIT:
    """Given a dataset."""

    def test_jit_compatible(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 3, 2, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
        ])
        dataset = make_dataset(recalls, pres)
        result_nojit = replagrank(dataset, min_lag=2, size=2)
        result_jit = jit(replagrank, static_argnames=("min_lag", "size"))(
            dataset, min_lag=2, size=2
        )
        np.testing.assert_allclose(
            np.array(result_nojit), np.array(result_jit), equal_nan=True
        )
