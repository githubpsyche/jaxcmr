"""Tests for jaxcmr.analyses.backreplagrank."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr.analyses.backreplagrank import (
    backreplagrank,
    subject_back_rep_lagrank,
    plot_back_rep_lagrank,
)
from jaxcmr.analyses.replagrank import replagrank
from jaxcmr.helpers import make_dataset


def _rep_dataset():
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
    return make_dataset(recalls, pres, subject=jnp.array([0, 0, 1, 1]))


class TestReversalChangesResults:
    """Given a dataset with repeated items."""

    def test_reversal_changes_results(self):
        """backreplagrank differs from forward replagrank."""
        dataset = _rep_dataset()
        forward = np.array(replagrank(dataset, min_lag=2, size=2))
        backward = np.array(backreplagrank(dataset, min_lag=2, size=2))
        # At least some finite values should differ
        both_finite = np.isfinite(forward) & np.isfinite(backward)
        if both_finite.any():
            assert not np.allclose(forward[both_finite], backward[both_finite])


class TestBackreplagrankIntegration:
    """Given a dataset with repeated items."""

    def test_backreplagrank_returns_shape(self):
        dataset = _rep_dataset()
        result = backreplagrank(dataset, min_lag=2, size=2)
        assert result.shape == (4, 2)


class TestSubjectBackRepLagrank:
    """Given a dataset with 2 subjects."""

    def test_subject_shape(self):
        dataset = _rep_dataset()
        mask = jnp.ones(4, dtype=bool)
        result = subject_back_rep_lagrank(dataset, mask, min_lag=2, size=2)
        assert result.shape == (2, 2)


class TestPlotBackRepLagrank:
    """Given a dataset and mask."""

    def test_plot_returns_axes(self):
        dataset = _rep_dataset()
        mask = jnp.ones(4, dtype=bool)
        ax = plot_back_rep_lagrank(dataset, mask, min_lag=2, size=2, labels=["1st", "2nd"])
        assert isinstance(ax, Axes)
