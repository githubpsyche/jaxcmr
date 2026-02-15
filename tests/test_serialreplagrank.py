"""Tests for jaxcmr.analyses.serialreplagrank."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.serialreplagrank import (
    SerialRepLagRankTabulation,
    serialreplagrank,
    subject_serial_rep_lagrank,
    plot_serial_rep_lagrank,
    test_serial_rep_lagrank_vs_control as run_test_vs_control,
    SerialRepLagRankTestResult,
)
from jaxcmr.helpers import make_dataset


# ---- Tabulation: only tabulates once ----

class TestTabulatesOnce:
    """Given a serial recall from a repeated item, only one transition is counted."""

    def test_only_tabulates_once(self):
        """At most 1 transition per trial because of has_tabulated flag."""
        # Item 1 at positions 1 and 6 (spacing=5 > min_lag=2).
        # Serial recall: 1,2,3,4,5,6 — position 1 first, then 2, etc.
        pres = jnp.array([1, 2, 3, 4, 5, 1])
        tab = SerialRepLagRankTabulation(pres, jnp.int32(1), min_lag=2, size=2)
        # Recall item 2 at output position 1 → study pos check: 2 == 1+1 ✓
        tab = tab.tabulate(jnp.int32(2), jnp.int32(1))
        count_after_first = int(tab.transition_count.sum())
        # Recall item 3 at output position 2 → study pos check: 3 == 2+1 ✓
        # But has_tabulated should be True, so no further tabulation.
        tab = tab.tabulate(jnp.int32(3), jnp.int32(2))
        count_after_second = int(tab.transition_count.sum())
        assert count_after_first == count_after_second


# ---- Tabulation: out-of-order recall is skipped ----

class TestOutOfOrderSkipped:
    """Given an out-of-order recall, future tabulations are blocked."""

    def test_out_of_order_sets_errored(self):
        """When recall breaks serial order, has_errored is set."""
        # Item 1 at positions 1 and 6 (spacing=5 > min_lag=2).
        pres = jnp.array([1, 2, 3, 4, 5, 1])
        tab = SerialRepLagRankTabulation(pres, jnp.int32(1), min_lag=2, size=2)
        # Recall item 3 at output position 1 → study pos check: 3 == 1+1? No → has_errored
        tab = tab.tabulate(jnp.int32(3), jnp.int32(1))
        assert bool(tab.has_errored)

    def test_prior_error_blocks_tabulation(self):
        """When has_errored was already set, no new transitions are counted."""
        # Item 1 at positions 1 and 8, item 2 at positions 2 and 7.
        pres = jnp.array([1, 2, 3, 4, 5, 6, 2, 1])
        tab = SerialRepLagRankTabulation(pres, jnp.int32(1), min_lag=2, size=2)
        # Recall item 3 at output 1 → breaks order → sets has_errored
        tab = tab.tabulate(jnp.int32(3), jnp.int32(1))
        count_after_error = int(tab.transition_count.sum())
        # Recall item 2 at output 2 → item 2 is repeated, but has_errored blocks
        tab = tab.tabulate(jnp.int32(2), jnp.int32(2))
        assert int(tab.transition_count.sum()) == count_after_error


# ---- Integration: shape and range ----

class TestSerialReplagrankIntegration:
    """Given a dataset with repeated items."""

    @pytest.fixture
    def dataset(self):
        # Item 1 at pos 1 and 6 (spacing=5), item 2 at pos 2 and 7 (spacing=5)
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 2, 3, 4, 0, 0, 0, 0],  # serial order, item 1 is repeated
            [3, 1, 2, 5, 0, 0, 0, 0],  # out of order
            [1, 2, 3, 5, 0, 0, 0, 0],  # serial order through pos 3
        ])
        return make_dataset(recalls, pres)

    def test_returns_shape(self, dataset):
        """serialreplagrank returns (n_trials, size)."""
        result = serialreplagrank(dataset, min_lag=2, size=2)
        assert result.shape == (3, 2)

    def test_values_in_range(self, dataset):
        """All finite values are in [0, 1]."""
        result = serialreplagrank(dataset, min_lag=2, size=2)
        finite = result[jnp.isfinite(result)]
        if finite.size > 0:
            assert float(finite.min()) >= 0.0
            assert float(finite.max()) <= 1.0

    def test_out_of_order_trial_is_nan(self, dataset):
        """A trial that breaks serial order should have NaN factors."""
        result = serialreplagrank(dataset, min_lag=2, size=2)
        # Trial 1 starts with item 3 (study pos 3), but recall_idx=0 → check 3==1? No → error
        assert bool(jnp.all(jnp.isnan(result[1])))


# ---- Subject aggregation ----

class TestSubjectSerialRepLagrank:
    """Given a dataset with 2 subjects."""

    def test_subject_shape(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 2, 3, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 0, 0, 0],
            [1, 2, 4, 3, 0, 0, 0, 0],
        ])
        subjects = jnp.array([0, 0, 1, 1])
        dataset = make_dataset(recalls, pres, subject=subjects)
        mask = jnp.ones(4, dtype=bool)
        result = subject_serial_rep_lagrank(dataset, mask, min_lag=2, size=2)
        assert result.shape == (2, 2)


# ---- Statistical tests ----

class TestSerialRepLagrankStats:
    """Given per-subject factor arrays."""

    def test_vs_control_returns_dict(self):
        rng = np.random.default_rng(42)
        observed = rng.uniform(0.4, 0.8, size=(20, 2))
        control = rng.uniform(0.3, 0.7, size=(20, 2))
        result = run_test_vs_control(observed, control)
        assert isinstance(result, dict)
        assert "First Presentation" in result
        assert "Second Presentation" in result
        assert isinstance(result["First Presentation"], SerialRepLagRankTestResult)


# ---- Plotting ----

class TestPlotSerialRepLagrank:
    """Given a dataset and mask."""

    def test_plot_returns_axes(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 2, 3, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
        ])
        dataset = make_dataset(recalls, pres)
        mask = jnp.ones(2, dtype=bool)
        ax = plot_serial_rep_lagrank(
            dataset, mask, min_lag=2, size=2, labels=["1st", "2nd"]
        )
        assert isinstance(ax, Axes)


# ---- JIT compatibility ----

class TestSerialReplagrankJIT:
    """Given a dataset."""

    def test_jit_compatible(self):
        pres = jnp.array([
            [1, 2, 3, 4, 5, 1, 2, 8],
            [1, 2, 3, 4, 5, 1, 2, 8],
        ])
        recalls = jnp.array([
            [1, 2, 3, 4, 0, 0, 0, 0],
            [1, 2, 3, 5, 0, 0, 0, 0],
        ])
        dataset = make_dataset(recalls, pres)
        result_nojit = serialreplagrank(dataset, min_lag=2, size=2)
        result_jit = jit(serialreplagrank, static_argnames=("min_lag", "size"))(
            dataset, min_lag=2, size=2
        )
        np.testing.assert_allclose(
            np.array(result_nojit), np.array(result_jit), equal_nan=True
        )
