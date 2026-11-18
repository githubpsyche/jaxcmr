"""Summarize participant scores and compare paired measurements.

``mean_interval`` reports a mean and bootstrap interval from independent
observations, usually one score per participant. ``compare_to_reference`` and
``compare_paired`` report means, differences, t-tests and Wilcoxon signed-rank
tests in DataFrames. These operations use finite observations and give each
included participant equal weight.

Calculate participant scores in the analysis module before using these
operations. Paired arrays must have the same participants in the same order.
Each comparison uses its own complete pairs, including for its marginal means.
The tests are two-sided and their p-values are not adjusted for multiplicity.
They answer different questions: the t-test concerns the mean difference;
the signed-rank test concerns a distribution of differences symmetric about zero.

These NumPy/SciPy reporting operations run outside JAX compilation.
"""

from collections.abc import Hashable, Iterable, Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from jaxcmr.typing import Float, Integer

__all__ = ["compare_paired", "compare_to_reference", "mean_interval"]

type _SeedLike = (
    int
    | np.integer
    | Sequence[int]
    | Integer[np.ndarray, "..."]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | None
)


class _MeanInterval(TypedDict):
    mean: float
    low: float
    high: float
    n: int
    confidence_level: float
    method: str
    n_resamples: int
    seed: _SeedLike


def mean_interval(
    values: ArrayLike,
    confidence_level: float = 0.95,
    method: str = "BCa",
    n_resamples: int = 9999,
    seed: _SeedLike = None,
) -> _MeanInterval:
    """Return the finite-observation mean and its bootstrap interval.

    Parameters
    ----------
    values : array_like
        Independent observations, usually participant scores. Nonfinite values
        are omitted; every remaining observation receives equal weight.
    confidence_level : float, optional
        Interval coverage, default 0.95.
    method : str, optional
        SciPy bootstrap method, default ``"BCa"``.
    n_resamples : int, optional
        Number of resamples, default 9999.
    seed : int or numpy.random.Generator, optional
        Random seed or generator. NumPy seed sequences and bit generators are
        also accepted. Omit for a newly initialized random generator.

    Returns
    -------
    dict
        ``mean``, ``low``, ``high``, finite sample size ``n``, and the supplied
        bootstrap settings. Fewer than two observations gives missing limits;
        an empty sample also has a missing mean. A constant sample of at least
        two observations has its point value as both limits.
    """
    samples = np.asarray(values, dtype=float)
    samples = samples[np.isfinite(samples)]
    mean = float(samples.mean()) if samples.size else float("nan")
    low = high = float("nan")
    if samples.size > 1:
        if np.ptp(samples) == 0:
            low = high = mean
        else:
            result = stats.bootstrap(
                (samples,),
                np.mean,
                confidence_level=confidence_level,
                method=method,
                n_resamples=n_resamples,
                rng=np.random.default_rng(seed),
            )
            low, high = map(float, result.confidence_interval)
    return _MeanInterval(
        mean=mean,
        low=low,
        high=high,
        n=int(samples.size),
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        seed=seed,
    )


def compare_paired(
    left: Float[np.ndarray, " participants"]
    | Float[np.ndarray, " participants comparisons"],
    right: Float[np.ndarray, " participants"]
    | Float[np.ndarray, " participants comparisons"],
    *,
    labels: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Compare aligned participant scores, separately for each column.

    Parameters
    ----------
    left, right : array of shape (participants,) or (participants, comparisons)
        Matching scores with identical participant order and shape. Each column
        uses only participants with finite values on both sides.
    labels : sequence, optional
        One label per comparison. Omit for an integer index starting at zero.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, indexed by ``comparison``. Columns are ``n``,
        ``mean_left``, ``mean_right``, ``mean_difference``, ``t_stat``,
        ``t_pvalue``, ``w_stat`` and ``w_pvalue``. Differences are left minus
        right. Means and both tests use the same finite pairs.

    Notes
    -----
    T-tests require two pairs. Wilcoxon uses SciPy's automatic method and
    ``zero_method="wilcox"``; an empty or all-zero difference sample has missing
    signed-rank statistics. Constant nonzero differences retain SciPy's t-test
    result. No rounding or multiplicity adjustment is applied.
    """
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim == 1:
        left_values = left_values[:, None]
        right_values = right_values[:, None]
    rows = []
    for left_column, right_column in zip(left_values.T, right_values.T, strict=True):
        paired = np.isfinite(left_column) & np.isfinite(right_column)
        observed, comparison = left_column[paired], right_column[paired]
        differences = observed - comparison
        count = differences.size
        t_stat = t_pvalue = w_stat = w_pvalue = float("nan")
        if count > 1:
            result = stats.ttest_1samp(differences, 0.0)
            t_stat, t_pvalue = map(float, np.asarray(result, dtype=float))
        if np.any(differences != 0):
            signed_rank = stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            w_stat, w_pvalue = map(float, np.asarray(signed_rank, dtype=float))
        rows.append(
            {
                "n": int(count),
                "mean_left": float(observed.mean()) if count else float("nan"),
                "mean_right": float(comparison.mean()) if count else float("nan"),
                "mean_difference": float(differences.mean()) if count else float("nan"),
                "t_stat": t_stat,
                "t_pvalue": t_pvalue,
                "w_stat": w_stat,
                "w_pvalue": w_pvalue,
            }
        )
    return pd.DataFrame(
        rows,
        index=pd.Index(
            labels if labels is not None else range(len(rows)), name="comparison"
        ),
        columns=pd.Index(
            [
                "n",
                "mean_left",
                "mean_right",
                "mean_difference",
                "t_stat",
                "t_pvalue",
                "w_stat",
                "w_pvalue",
            ]
        ),
    )


def compare_to_reference(
    values: Float[np.ndarray, " participants"]
    | Float[np.ndarray, " participants comparisons"],
    reference: float = 0.0,
    *,
    labels: Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Compare participant scores with a constant reference value.

    Parameters
    ----------
    values : array of shape (participants,) or (participants, comparisons)
        Independent participant scores, optionally for several comparisons.
    reference : float, optional
        Null value, such as 0.5 for a rank score or zero for a slope.
    labels : sequence, optional
        One label per comparison.

    Returns
    -------
    pandas.DataFrame
        The same columns and finite-observation conventions as
        ``compare_paired``. ``mean_left`` is the score mean and ``mean_right``
        is the reference for each nonempty comparison.
    """
    samples = np.asarray(values, dtype=float)
    return compare_paired(samples, np.full_like(samples, reference), labels=labels)
