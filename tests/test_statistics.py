"""Interval configuration and repeatability against SciPy."""
import numpy as np
from scipy.stats import bootstrap
import pytest
from jaxcmr.analyses.statistics import mean_interval, compare_paired, compare_to_reference


def test_bootstrap_configuration_is_explicit_and_reproducible():
    values = np.array([0.11, 0.25, 0.6, 0.9, 0.93])
    result = mean_interval(values, seed=19, n_resamples=399)
    expected = bootstrap((values,), np.mean, method='BCa', n_resamples=399,
                         rng=np.random.default_rng(19)).confidence_interval
    np.testing.assert_allclose([result['low'], result['high']], expected)
    assert result == mean_interval(values, seed=19, n_resamples=399)
    assert mean_interval([0.5, 0.5])['low'] == 0.5


def test_paired_statistics_use_the_same_finite_pairs_for_every_summary():
    left = np.array([[.8, 1], [.9, np.nan], [.7, 3], [99, np.inf]])
    right = np.array([[.3, 0], [.2, 4], [.4, 1], [np.nan, 2]])
    result = compare_paired(left, right, labels=["response", "other"])
    row = result.loc["response"]
    # Differences .5,.7,.3: mean .5, sample SD .2, two degrees of freedom.
    t = .5 * np.sqrt(3) / .2
    assert row["n"] == 3
    assert row["mean_left"] == pytest.approx(.8)
    assert row["mean_right"] == pytest.approx(.3)
    assert row["mean_difference"] == pytest.approx(.5)
    assert row["t_stat"] == pytest.approx(t)
    assert row["t_pvalue"] == pytest.approx(1 - t / np.sqrt(t*t + 2))
    assert row["w_stat"] == 0
    assert row["w_pvalue"] == .25
    assert result.loc["other", "n"] == 2


def test_reference_comparison_and_undefined_small_samples():
    result = compare_to_reference(np.array([[.5, .8, np.nan], [.5, np.nan, np.inf]]), .5)
    assert result.loc[0, "mean_difference"] == 0
    assert np.isnan(result.loc[0, "w_pvalue"])
    assert result.loc[1, "n"] == 1
    assert np.isnan(result.loc[1, "t_stat"])
    assert result.loc[2, "n"] == 0
    assert np.isnan(result.loc[2, "mean_left"])
    single = mean_interval([.3, np.inf, np.nan])
    assert single["mean"] == .3 and single["n"] == 1
    assert np.isnan(single["low"]) and np.isnan(single["high"])
