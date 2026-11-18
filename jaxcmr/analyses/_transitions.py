"""Shared array calculations for transition tabulators.

These operations update availability, count signed lags, or rank an eligible
alternative. The calling analysis owns transition eligibility and history.
"""

from jax import lax
from jax import numpy as jnp

from ..typing import Array, Bool, Float, Int_, Integer, Real


def remove_positions(
    available: Bool[Array, " positions"],
    positions: Integer[Array, " occurrences"],
) -> Bool[Array, " positions"]:
    """Remove one-based positions from availability; zero is an unused entry."""

    def remove(
        mask: Bool[Array, " positions"], position: Int_
    ) -> tuple[Bool[Array, " positions"], None]:
        included = (position > 0) & (position <= mask.size)
        return lax.cond(
            included, lambda: mask.at[position - 1].set(False), lambda: mask
        ), None

    return lax.scan(remove, available, positions)[0]


def lag_counts(
    origin: Int_,
    targets: Integer[Array, " targets"],
    eligible: Bool[Array, " targets"],
    max_lag: int,
) -> Integer[Array, " lag_bins"]:
    """Count eligible targets by signed target-minus-origin lag.

    Positions are one-based; zero positions contribute nothing. Exclude lags
    outside the inclusive [-max_lag, max_lag] window. ``max_lag`` is static.
    Callers convert counts to Boolean unions when their estimator requires it.
    """
    lags = targets - origin
    include = eligible & (origin > 0) & (targets > 0) & (jnp.abs(lags) <= max_lag)
    indices = jnp.clip(lags + max_lag, 0, 2 * max_lag)
    return jnp.zeros(2 * max_lag + 1, dtype=int).at[indices].add(include.astype(int))


def absolute_lag_counts(
    origin: Int_,
    targets: Integer[Array, " targets"],
    eligible: Bool[Array, " targets"],
    max_lag: int,
) -> Integer[Array, " absolute_lag_bins"]:
    """Fold signed lag counts into bins 0..max_lag for historical lag ranks.

    Left and right alternatives at the same distance contribute to one bin.
    Callers decide whether to retain counts or use the presence of each bin.
    """
    signed = lag_counts(origin, targets, eligible, max_lag)
    return signed[max_lag:].at[1:].add(signed[:max_lag][::-1])


def masked_midrank(
    target: Real[Array, ""],
    alternatives: Real[Array, " alternatives"],
    available: Bool[Array, " alternatives"],
) -> Float[Array, ""]:
    """Rank a selected value from one (smallest) to zero (largest), with ties.

    Use midpoint ranks among available alternatives, including the selected
    value itself. Fewer than two alternatives gives NaN. The caller determines
    availability; this does not implement historical distinct-lag-bin ranks.
    """
    count = jnp.sum(available)
    smaller = jnp.sum(available & (alternatives < target))
    equal = jnp.sum(available & (alternatives == target))
    rank = (count - 1 - smaller - 0.5 * (equal - 1)) / (count - 1)
    return lax.select(count > 1, rank, jnp.full_like(rank, jnp.nan))
