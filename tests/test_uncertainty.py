"""Interval configuration and repeatability against SciPy."""
import numpy as np
from scipy.stats import bootstrap
from jaxcmr.analyses.uncertainty import mean_interval


def test_bootstrap_configuration_is_explicit_and_reproducible():
    values = np.array([0.11, 0.25, 0.6, 0.9, 0.93])
    result = mean_interval(values, seed=19, n_resamples=399)
    expected = bootstrap((values,), np.mean, method='BCa', n_resamples=399,
                         rng=np.random.default_rng(19)).confidence_interval
    np.testing.assert_allclose([result['low'], result['high']], expected)
    assert result == mean_interval(values, seed=19, n_resamples=399)
    assert mean_interval([0.5, 0.5])['low'] == 0.5
