"""Explicit, reproducible participant-bootstrap intervals."""
import numpy as np
from scipy.stats import bootstrap


def mean_interval(values, confidence_level=0.95, method="BCa", n_resamples=9999, seed=None):
    """Summarize finite independent-unit values, not individual transitions.

    A constant sample has its point value as both limits. Fewer than two finite
    units gives missing limits. ``seed=None`` preserves SciPy's default RNG.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if values.size else float('nan')
    low = high = float('nan')
    if values.size > 1:
        if np.ptp(values) == 0:
            low = high = mean
        else:
            result = bootstrap(
                (values,), np.mean, confidence_level=confidence_level,
                method=method, n_resamples=n_resamples,
                rng=np.random.default_rng(seed) if seed is not None else None,
            )
            low, high = map(float, result.confidence_interval)
    return dict(mean=mean, low=low, high=high, n=int(values.size),
                confidence_level=confidence_level, method=method,
                n_resamples=n_resamples, seed=seed)
