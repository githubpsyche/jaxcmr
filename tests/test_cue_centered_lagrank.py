"""Tests for jaxcmr.analyses.cue_centered_lagrank."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp

from jaxcmr.analyses.cue_centered_lagrank import (
    CueCenteredLagRankTabulation,
    cue_centered_lagrank,
    subject_cue_centered_lagrank,
    plot_cue_centered_lagrank,
    test_cue_centered_lagrank as run_test,
)
from jaxcmr.analyses.lagrank import LagRankTestResult
from jaxcmr.helpers import make_dataset


# ---- Tabulation: no cue → skipped ----

class TestNoCueSkipped:
    """Given a recall event with no cue, no transition is counted."""

    def test_no_cue_skipped(self):
        """When cue is 0 for all events, transition_count stays zero."""
        pres = jnp.array([1, 2, 3, 4])
        tab = CueCenteredLagRankTabulation(pres, size=1)
        # Recall item 1 with no cue
        tab = tab.tabulate(jnp.int32(1), jnp.int32(0), jnp.bool_(True))
        assert int(tab.transition_count) == 0


# ---- Tabulation: valid cue → tabulated ----

class TestValidCueTabulated:
    """Given a recall event with a valid cue, the transition is counted."""

    def test_valid_cue_tabulated(self):
        """When cue is valid and should_tabulate is True, count increases."""
        pres = jnp.array([1, 2, 3, 4])
        tab = CueCenteredLagRankTabulation(pres, size=1)
        # Recall item 2, cued by item 1
        tab = tab.tabulate(jnp.int32(2), jnp.int32(1), jnp.bool_(True))
        assert int(tab.transition_count) > 0

    def test_should_tabulate_false_skips(self):
        """When should_tabulate is False, nothing is counted."""
        pres = jnp.array([1, 2, 3, 4])
        tab = CueCenteredLagRankTabulation(pres, size=1)
        tab = tab.tabulate(jnp.int32(2), jnp.int32(1), jnp.bool_(False))
        assert int(tab.transition_count) == 0


# ---- Integration: shape and range ----

class TestCueCenteredLagrankIntegration:
    """Given a dataset with cues."""

    @pytest.fixture
    def dataset(self):
        pres = jnp.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        recalls = jnp.array([
            [1, 2, 3, 0],
            [2, 3, 0, 0],
            [1, 3, 4, 0],
        ])
        cues = jnp.array([
            [1, 1, 2, 0],
            [1, 2, 0, 0],
            [3, 2, 1, 0],
        ])
        should_tab = recalls > 0
        return {**make_dataset(recalls, pres), "cue_clips": cues, "_should_tabulate": should_tab}

    def test_returns_1d(self, dataset):
        """cue_centered_lagrank returns (n_trials,)."""
        result = cue_centered_lagrank(dataset, size=1)
        assert result.shape == (3,)

    def test_values_in_range(self, dataset):
        """All finite values are in [0, 1]."""
        result = cue_centered_lagrank(dataset, size=1)
        finite = result[jnp.isfinite(result)]
        if finite.size > 0:
            assert float(finite.min()) >= 0.0
            assert float(finite.max()) <= 1.0


# ---- Subject aggregation ----

class TestSubjectCueCenteredLagrank:
    """Given a dataset with 2 subjects."""

    def test_subject_shape(self):
        pres = jnp.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        recalls = jnp.array([
            [1, 2, 3, 0],
            [2, 3, 0, 0],
            [1, 3, 4, 0],
            [2, 1, 0, 0],
        ])
        cues = jnp.array([
            [1, 1, 2, 0],
            [1, 2, 0, 0],
            [3, 2, 1, 0],
            [1, 3, 0, 0],
        ])
        should_tab = recalls > 0
        subjects = jnp.array([0, 0, 1, 1])
        dataset = {**make_dataset(recalls, pres, subject=subjects), "cue_clips": cues, "_should_tabulate": should_tab}
        mask = jnp.ones(4, dtype=bool)
        result = subject_cue_centered_lagrank(dataset, mask, size=1)
        assert result.shape == (2,)


# ---- Statistical test ----

class TestCueCenteredLagrankStats:
    """Given per-subject factor array."""

    def test_against_chance(self):
        rng = np.random.default_rng(42)
        factors = rng.uniform(0.6, 0.9, size=20)
        result = run_test(factors)
        assert isinstance(result, LagRankTestResult)
        assert result.n == 20


# ---- Plotting ----

class TestPlotCueCenteredLagrank:
    """Given a dataset and mask."""

    def test_plot_returns_axes(self):
        pres = jnp.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        recalls = jnp.array([
            [1, 2, 3, 0],
            [2, 3, 0, 0],
        ])
        cues = jnp.array([
            [1, 1, 2, 0],
            [1, 2, 0, 0],
        ])
        should_tab = recalls > 0
        dataset = {**make_dataset(recalls, pres), "cue_clips": cues, "_should_tabulate": should_tab}
        mask = jnp.ones(2, dtype=bool)
        ax = plot_cue_centered_lagrank(
            dataset, mask, should_tabulate=should_tab, size=1, labels=["Test"]
        )
        from matplotlib.axes import Axes
        assert isinstance(ax, Axes)


# ---- JIT compatibility ----

class TestCueCenteredLagrankJIT:
    """Given a dataset."""

    def test_jit_compatible(self):
        pres = jnp.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        recalls = jnp.array([
            [1, 2, 3, 0],
            [2, 3, 0, 0],
        ])
        cues = jnp.array([
            [1, 1, 2, 0],
            [1, 2, 0, 0],
        ])
        should_tab = recalls > 0
        dataset = {**make_dataset(recalls, pres), "cue_clips": cues, "_should_tabulate": should_tab}
        result_nojit = cue_centered_lagrank(dataset, size=1)
        result_jit = jit(cue_centered_lagrank, static_argnames=("size",))(
            dataset, size=1
        )
        np.testing.assert_allclose(
            np.array(result_nojit), np.array(result_jit), equal_nan=True
        )
