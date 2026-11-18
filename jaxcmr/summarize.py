"""Summarize fitted parameters and compare models using their scores.

Use the numerical functions, such as ``aic_scores``, when you have arrays
of scores and want to work with the results directly. When you have named
model records, the comparison functions align rows by participant ID and
produce labelled tables. Which observations were fitted and what a score
measures must agree across the models in a comparison.

``summarize_parameters`` describes the distribution of parameter estimates
across fitted rows. For comparing models, information criteria start with
summed negative log likelihoods and penalize the number of free parameters.
Paired differences and win fractions describe how the models compare for
the same participants. Held-out comparisons use the supplied test scores.
For declared nested models, ``floor_nested_fitness`` makes a reporting
adjustment and records which scores it changed.

To estimate model frequencies in a population, ``random_effects_model_selection``
takes one row of log evidences per participant, with one column per model.
``bayesian_model_selection`` prepares that comparison from fitted likelihoods
using an information-criterion approximation. A single fit to pooled
observations supplies only one estimate, regardless of the number of
participants who contributed trials.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple, NotRequired, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import t, ttest_rel
from typing_extensions import ReadOnly, TypedDict

from jaxcmr.fitting.parameters import bound_params
from jaxcmr.typing import Array, Float, Integer

__all__ = [
    "CVComparison",
    "ModelComparison",
    "ObjectiveSummary",
    "ParameterSummary",
    "add_summary_lines",
    "aic_scores",
    "aicc_scores",
    "aicc_winner_comparison_matrix",
    "bayesian_model_selection",
    "bic_scores",
    "bound_params",
    "calculate_aic",
    "calculate_aic_weights",
    "calculate_aicc",
    "calculate_aicc_weights",
    "calculate_bic_scores",
    "calculate_ci",
    "cv_winner_comparison_matrix",
    "floor_nested_fitness",
    "generate_t_p_matrices",
    "pairwise_aic_differences",
    "pairwise_aicc_differences",
    "pairwise_cv_differences",
    "pairwise_median_aic_differences",
    "pairwise_median_aicc_differences",
    "random_effects_model_selection",
    "raw_winner_comparison_matrix",
    "summarize_parameters",
    "winner_comparison_matrix",
]


type ScoreValues = Float[Array | np.ndarray, " fit_rows"] | Sequence[float]
type SubjectIds = Integer[Array | np.ndarray, " fit_rows"] | Sequence[int]
type Criterion = Literal["AIC", "AICc", "BIC"]


class ObjectiveSummary(TypedDict):
    """Named objectives aligned to a top-level integer ``subject`` vector.

    Comparisons require nonempty finite scores, distinct model names, and unique
    IDs covering the same participant set. These are caller preconditions.
    """

    name: ReadOnly[str]
    subject: ReadOnly[SubjectIds]
    fitness: ReadOnly[ScoreValues]


class ParameterSummary(TypedDict):
    """Objective values and parameter columns for a descriptive table.

    Columns may contain NaNs. The reserved ``fits["subject"]`` column is omitted
    from the table; parameter summaries do not pair participants across models.
    """

    name: ReadOnly[str]
    fitness: ReadOnly[ScoreValues]
    fits: ReadOnly[Mapping[str, ArrayLike]]


class ModelComparison(TypedDict):
    """Named NLL scores, integer row IDs, and free-parameter names.

    Supply a top-level ``subject`` vector aligned with ``fitness``; use ``[-1]``
    for a single pooled fit. Participant comparisons require the same unique
    participant IDs across models and exclude pooled fits. Scores must be finite,
    names distinct, and ``free`` must list distinct names rather than bounds.
    These are caller preconditions, not runtime validation.
    """

    name: ReadOnly[str]
    subject: ReadOnly[SubjectIds]
    fitness: ReadOnly[ScoreValues]
    free: ReadOnly[Sequence[str]]
    comparison_only: NotRequired[ReadOnly[bool]]


class CVComparison(TypedDict):
    """Held-out objectives aligned to a top-level integer ``subject`` vector.

    Supply distinct names, finite scores, and the same unique participant IDs
    across models. Pooled fits are not participant samples.
    """

    name: ReadOnly[str]
    subject: ReadOnly[SubjectIds]
    cv_fitness: ReadOnly[ScoreValues]


class _AlignedScores(NamedTuple):
    """Scores in common row order and the first record's row permutation."""

    model_names: list[str]
    scores: Float[np.ndarray, "models fit_rows"]
    subject_ids: list[int]
    first_record_order: Integer[np.ndarray, " fit_rows"]


def _nonmissing_values(values: ArrayLike) -> Float[np.ndarray, " observations"]:
    """Omit missing observations for all descriptive statistics alike."""
    variant_array = np.asarray(values, dtype=float)
    return variant_array[~np.isnan(variant_array)]


def calculate_ci(data: ArrayLike, confidence: float = 0.95) -> float:
    """Return the two-sided Student-t confidence-interval half-width.

    Parameters
    ----------
    data : array_like
        Observations to summarize. NaNs are omitted; infinities are not missing.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.

    Returns
    -------
    float
        Interval half-width, or NaN with fewer than two nonmissing observations.

    """
    values = _nonmissing_values(data)
    sample_count = len(values)
    if sample_count < 2:
        return float("nan")
    standard_error = np.std(values, ddof=1) / np.sqrt(sample_count)
    return float(standard_error * t.ppf((1 + confidence) / 2.0, sample_count - 1))


def _format_row(parameter_label: str, statistic_label: str, values: list[str]) -> str:
    """Format one Markdown summary-table row.

    Parameters
    ----------
    parameter_label : str
        Parameter or metric name, or an empty string for a continuation row.
    statistic_label : str
        Statistic name for this row.
    values : list[str]
        Formatted values in model-column order.

    Returns
    -------
    str
        Markdown row with a trailing newline.

    """
    cells = [parameter_label, statistic_label, *values]
    sanitized = [cell or "" for cell in cells]
    return "| " + " | ".join(sanitized) + " |\n"


def _format_mean_cell(
    array: Float[np.ndarray, " observations"], include_ci: bool
) -> str:
    """Format a mean with optional confidence-interval text.

    Parameters
    ----------
    array : Float[np.ndarray, " observations"]
        Observations with missing values already omitted.
    include_ci : bool
        Whether to append the 95% Student-t interval half-width.

    Returns
    -------
    str
        Mean rounded to two decimal places, with an interval when requested.
        Empty or undefined means produce an empty cell; an undefined interval
        is labelled ``CI unavailable``.

    """
    if array.size == 0:
        return ""
    mean_value = np.mean(array)
    if np.isnan(mean_value):
        return ""
    if include_ci:
        halfwidth = calculate_ci(array)
        if np.isnan(halfwidth):
            return f"{mean_value:.2f} (CI unavailable)"
        return f"{mean_value:.2f} +/- {halfwidth:.2f}"
    return f"{mean_value:.2f}"


def _format_stat_cell(
    array: Float[np.ndarray, " observations"],
    reducer: Callable[[Float[np.ndarray, " observations"]], float],
) -> str:
    """Format a scalar descriptive statistic.

    Parameters
    ----------
    array : Float[np.ndarray, " observations"]
        Observations with missing values already omitted.
    reducer : callable
        Function reducing the observations to one numerical value.

    Returns
    -------
    str
        Statistic rounded to two decimal places, or an empty cell for an empty
        sample or a NaN result.

    """
    if array.size == 0:
        return ""
    value = reducer(array)
    return "" if np.isnan(value) else f"{value:.2f}"


def add_summary_lines(
    md_table: str,
    errors: Sequence[ArrayLike],
    label: str,
    include_std: bool = False,
    include_ci: bool = False,
) -> str:
    """Append descriptive-statistic rows to a Markdown table.

    Parameters
    ----------
    md_table : str
        Existing table fragment.
    errors : Sequence[array_like]
        Observations for each model variant, in table-column order.
    label : str
        Parameter or metric name; underscores are displayed as spaces.
    include_std : bool, optional
        Include a population-standard-deviation row (ddof=0); default is False.
    include_ci : bool, optional
        Append 95% Student-t interval half-widths to means; default is False.

    Returns
    -------
    str
        Table with mean, optional standard deviation, minimum and maximum rows.

    Notes
    -----
    NaNs are omitted consistently across statistics. Empty samples produce blank
    cells. Means with fewer than two observations show ``CI unavailable`` when
    intervals are requested.

    """
    display_label = label.replace("_", " ")
    variant_arrays = [_nonmissing_values(values) for values in errors]

    mean_values = [
        _format_mean_cell(variant_array, include_ci) for variant_array in variant_arrays
    ]
    md_table += _format_row(display_label, "mean", mean_values)

    if include_std:
        std_values = [
            _format_stat_cell(variant_array, np.std) for variant_array in variant_arrays
        ]
        md_table += _format_row("", "std", std_values)

    min_values = [
        _format_stat_cell(variant_array, np.min) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "min", min_values)

    max_values = [
        _format_stat_cell(variant_array, np.max) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "max", max_values)

    return md_table


def summarize_parameters(
    model_data: Sequence[ParameterSummary],
    query_parameters: list[str] | None = None,
    include_std: bool = False,
    include_ci: bool = False,
) -> str:
    """Build a Markdown table of parameter statistics across models.

    Parameters
    ----------
    model_data : Sequence[ParameterSummary]
        Named models with objective values in ``fitness`` and parameter columns
        in ``fits``. The input records are not modified.
    query_parameters : list[str] or None, optional
        Ordered parameter names. None selects all ``fits`` columns except
        ``subject``, in first-seen order. Missing columns produce blank cells.
    include_std : bool, optional
        Include population standard deviations (ddof=0); default is False.
    include_ci : bool, optional
        Append 95% Student-t interval half-widths to means; default is False.

    Returns
    -------
    str
        Markdown table with Parameter and Statistic columns followed by one
        column per model. Objective statistics precede parameter statistics.

    Notes
    -----
    Each sample omits NaNs. Means with fewer than two nonmissing observations
    show ``CI unavailable`` when intervals are requested. These descriptive
    summaries do not pair participants across models.

    """
    if query_parameters is None:
        # Dictionaries preserve the first occurrence of each parameter name.
        parameter_names: dict[str, None] = {}
        for model in model_data:
            parameter_names |= dict.fromkeys(model["fits"])
        parameter_names.pop("subject", None)
        query_parameters = list(parameter_names)
    header_names = [model["name"].replace("_", " ") for model in model_data]
    table = "| Parameter | Statistic | " + " | ".join(header_names) + " |\n"
    table += "|---|---" + "|---" * len(model_data) + "|\n"
    table = add_summary_lines(
        table,
        [model["fitness"] for model in model_data],
        "fitness",
        include_std=include_std,
        include_ci=include_ci,
    )
    for parameter in query_parameters:
        table = add_summary_lines(
            table,
            [model["fits"].get(parameter, []) for model in model_data],
            parameter,
            include_std=include_std,
            include_ci=include_ci,
        )
    return table


def aic_scores(
    negative_log_likelihoods: ArrayLike, parameter_counts: ArrayLike
) -> Float[np.ndarray, "..."]:
    """Calculate elementwise AIC, ``2 * NLL + 2 * k``.

    Parameters
    ----------
    negative_log_likelihoods : array_like
        Summed negative log likelihood for each fitted parameter set.
        MSE and mean or scaled NLL are outside this interface.
    parameter_counts : array_like
        Nonnegative integer free-parameter counts, broadcastable to the NLLs.

    Returns
    -------
    np.ndarray
        Scores in broadcast input order, without sorting or aggregation.
        Scalar inputs produce a zero-dimensional array.

    Notes
    -----
    For NLLs shaped (models, participants), use parameter counts shaped
    (models, 1). Each fitted row incurs its own penalty. Inputs are unchanged.

    Examples
    --------
    >>> aic_scores([33.0, 30.0], [2, 1])
    array([70., 62.])

    """
    negative_log_likelihoods = np.asarray(negative_log_likelihoods, dtype=float)
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    return np.asarray(2 * negative_log_likelihoods + 2 * parameter_counts)


def aicc_scores(
    negative_log_likelihoods: ArrayLike,
    parameter_counts: ArrayLike,
    observation_counts: ArrayLike,
) -> Float[np.ndarray, "..."]:
    """Calculate elementwise AICc, ``AIC + 2*k*(k+1)/(n-k-1)``.

    Parameters
    ----------
    negative_log_likelihoods : array_like
        Summed negative log likelihood for each fitted parameter set.
    parameter_counts : array_like
        Nonnegative integer free-parameter counts.
    observation_counts : array_like
        Positive integer observation counts aligned with the scores. The caller
        defines the observation unit and whether this correction is appropriate.

    Returns
    -------
    np.ndarray
        Scores following NumPy broadcasting, without sorting or aggregation.
        Scalar inputs produce a zero-dimensional array. Inputs are unchanged.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    See Also
    --------
    aic_scores : NLL interpretation and broadcasting conventions.

    """
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    observation_counts = np.asarray(observation_counts, dtype=float)
    denominator = observation_counts - parameter_counts - 1
    if np.any(denominator <= 0):
        raise ValueError("AICc undefined where n_obs <= k + 1")
    return np.asarray(
        aic_scores(negative_log_likelihoods, parameter_counts)
        + 2 * parameter_counts * (parameter_counts + 1) / denominator
    )


def bic_scores(
    negative_log_likelihoods: ArrayLike,
    parameter_counts: ArrayLike,
    observation_counts: ArrayLike,
) -> Float[np.ndarray, "..."]:
    """Calculate elementwise BIC, ``2 * NLL + k * log(n)``.

    Parameters
    ----------
    negative_log_likelihoods : array_like
        Summed negative log likelihood for each fitted parameter set.
    parameter_counts : array_like
        Nonnegative integer free-parameter counts.
    observation_counts : array_like
        Positive integer observation counts aligned with the scores. The caller
        defines the observation unit; participant count is not substituted for it.

    Returns
    -------
    np.ndarray
        Scores following NumPy broadcasting, without sorting or aggregation.
        Scalar inputs produce a zero-dimensional array. Inputs are unchanged.

    See Also
    --------
    aic_scores : NLL interpretation and broadcasting conventions.

    """
    negative_log_likelihoods = np.asarray(negative_log_likelihoods, dtype=float)
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    observation_counts = np.asarray(observation_counts, dtype=float)
    return np.asarray(
        2 * negative_log_likelihoods + parameter_counts * np.log(observation_counts)
    )


def _align_model_scores(
    results: Sequence[ObjectiveSummary | CVComparison],
    score_rows: Sequence[ScoreValues],
) -> _AlignedScores:
    """Sort each model by its supplied IDs, retaining the first row permutation.

    All records must identify the same unique IDs and align their score rows
    with those IDs. Pooled comparisons use the single ID -1 in every record.
    """
    row_orders = [np.argsort(model["subject"]) for model in results]
    aligned_scores = np.stack(
        [
            np.asarray(scores, dtype=float)[order]
            for scores, order in zip(score_rows, row_orders)
        ]
    )
    subject_ids = np.asarray(results[0]["subject"])[row_orders[0]].tolist()
    return _AlignedScores(
        [model["name"] for model in results],
        aligned_scores,
        subject_ids,
        row_orders[0],
    )


def _aligned_criterion_scores(
    results: Sequence[ModelComparison],
    criterion: Criterion = "AIC",
    n_obs: ArrayLike | None = None,
) -> _AlignedScores:
    """Calculate per-fit criteria after pairing model rows by participant ID."""
    comparison = _align_model_scores(
        results,
        [model["fitness"] for model in results],
    )
    parameter_counts = np.array([len(model["free"]) for model in results])[:, None]
    if criterion == "AIC":
        scores = aic_scores(comparison.scores, parameter_counts)
    else:
        # Counts arrive in the first record's order, just like its score rows.
        observation_counts = np.atleast_1d(np.asarray(cast(ArrayLike, n_obs), dtype=float))
        observation_counts = observation_counts[comparison.first_record_order]
        if criterion == "BIC":
            scores = bic_scores(comparison.scores, parameter_counts, observation_counts)
        else:
            scores = aicc_scores(comparison.scores, parameter_counts, observation_counts)
    return comparison._replace(scores=scores)


def _criterion_table(
    results: Sequence[ModelComparison],
    criterion: Criterion,
    n_obs: ArrayLike | None = None,
) -> pd.DataFrame:
    """Build a ranked table of criteria summed across fitted rows."""
    comparison = _aligned_criterion_scores(results, criterion, n_obs)
    return pd.DataFrame(
        {"Model": comparison.model_names, criterion: comparison.scores.sum(axis=1)}
    ).sort_values(criterion)


def calculate_aic(results: Sequence[ModelComparison]) -> pd.DataFrame:
    """Return summed AIC scores, lowest first.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named records with top-level integer ``subject`` IDs, distinct ``free``
        parameter names and a one-dimensional ``fitness`` array. Each fitness
        value is summed NLL for one independently fitted parameter set.

    Returns
    -------
    pd.DataFrame
        Model and AIC columns, sorted by increasing AIC. Each model's score is
        ``sum_rows(2*NLL + 2*k)``.

    Notes
    -----
    A pooled fit uses ``subject=[-1]`` and incurs one parameter penalty.
    Participant fits require the same unique IDs across models and incur one
    penalty per fitted row. Names must be distinct and scores finite.

    MSE and mean or scaled NLL cannot substitute for summed NLL. The objective
    and its sign cannot be inferred from numerical values. These conventions
    also apply to AICc, BIC and information-criterion weights.

    """
    return _criterion_table(results, "AIC")


def calculate_bic_scores(
    results: Sequence[ModelComparison], n_obs: ArrayLike
) -> pd.DataFrame:
    """Return summed BIC scores, lowest first.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records following ``calculate_aic``.
    n_obs : array_like
        Positive integer observation counts in the first record's original
        row order. Counts are aligned with its participant IDs. A scalar is
        allowed only for one fitted row.

    Returns
    -------
    pd.DataFrame
        Model and BIC columns, sorted by increasing ``sum_rows(2*NLL + k*log(n))``.

    Notes
    -----
    The caller defines the scientific observation unit. The number of fitted
    participants is never substituted for observation counts.

    """
    return _criterion_table(results, "BIC", n_obs)


def calculate_aicc(
    results: Sequence[ModelComparison], n_obs: ArrayLike
) -> pd.DataFrame:
    """Return summed AICc scores, lowest first.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records following ``calculate_aic``.
    n_obs : array_like
        Positive integer observation counts in the first record's original
        row order. A scalar is allowed only for one fitted row. The caller
        defines the counting unit and the correction's scientific applicability.

    Returns
    -------
    pd.DataFrame
        Model and AICc columns, sorted by increasing
        ``sum_rows(AIC + 2*k*(k+1)/(n-k-1))``.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    """
    return _criterion_table(results, "AICc", n_obs)


def _criterion_weights(
    table: pd.DataFrame, criterion: str, column: str
) -> pd.DataFrame:
    """Normalize exponentiated score differences and rank model weights."""
    delta = table[criterion] - table[criterion].min()
    weights = np.exp(-0.5 * delta)
    return (
        table[["Model"]]
        .assign(**{column: weights / weights.sum()})
        .sort_values(column, ascending=False)
    )


def calculate_aic_weights(results: Sequence[ModelComparison]) -> pd.DataFrame:
    """Return normalized AIC weights, largest first.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records following ``calculate_aic``.

    Returns
    -------
    pd.DataFrame
        Model and AICw columns. Weights normalize ``exp(-0.5 * delta_AIC)``
        across models using their summed AIC scores.

    """
    return _criterion_weights(calculate_aic(results), "AIC", "AICw")


def calculate_aicc_weights(
    results: Sequence[ModelComparison], n_obs: ArrayLike
) -> pd.DataFrame:
    """Return normalized AICc weights, largest first.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records following ``calculate_aic``.
    n_obs : array_like
        Observation counts in the first record's original row order, following
        ``calculate_aicc``. A scalar is allowed only for one fitted row.

    Returns
    -------
    pd.DataFrame
        Model and AICcw columns. Weights normalize ``exp(-0.5 * delta_AICc)``
        across models using their summed AICc scores.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    """
    return _criterion_weights(calculate_aicc(results, n_obs), "AICc", "AICcw")


def _mean_differences(
    names: list[str],
    values: Float[np.ndarray, "models participants"],
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabulate paired mean differences and t-interval half-widths."""
    labels = pd.Index(names)
    means = pd.DataFrame(np.nan, index=labels, columns=labels)
    halfwidths = means.copy()
    if values.shape[1] < 2:
        return means, halfwidths
    for row, column in itertools.combinations(range(len(names)), 2):
        differences = values[row] - values[column]
        mean_difference = np.mean(differences)
        interval_halfwidth = calculate_ci(differences, confidence)
        means.iat[row, column] = mean_difference
        means.iat[column, row] = -mean_difference
        halfwidths.iat[row, column] = interval_halfwidth
        halfwidths.iat[column, row] = interval_halfwidth
    return means, halfwidths


def pairwise_aic_differences(
    results: Sequence[ModelComparison], confidence: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare mean participant-wise AIC differences with t intervals.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records following ``calculate_aic``. Each record must
        identify the same unique integer participants in its top-level ``subject``
        vector, aligned with finite scores. Pooled fits are excluded.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.

    Returns
    -------
    means : pd.DataFrame
        Mean row-model minus column-model differences. Negative favors the row
        model. Diagonals and comparisons with fewer than two participants are NaN.
    halfwidths : pd.DataFrame
        Two-sided Student-t interval half-widths with the same labels and NaNs.
        No Boolean equivalence decision is returned.

    Notes
    -----
    Rows are paired by participant ID, not input position. IDs, distinct model
    names and finite scores are caller preconditions. String and floating-point
    IDs are not normalized; the pooled sentinel -1 is not a participant sample.

    """
    comparison = _aligned_criterion_scores(results)
    return _mean_differences(comparison.model_names, comparison.scores, confidence)


def pairwise_aicc_differences(
    results: Sequence[ModelComparison],
    n_obs: ArrayLike,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare mean participant-wise AICc differences with t intervals.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant fits following ``pairwise_aic_differences``.
    n_obs : array_like
        Observation counts in the first record's original participant order,
        following ``calculate_aicc``.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.

    Returns
    -------
    means : pd.DataFrame
        Mean row-minus-column AICc differences; negative favors the row model.
        Diagonals and comparisons with fewer than two participants are NaN.
    halfwidths : pd.DataFrame
        Two-sided Student-t interval half-widths, with matching labels and NaNs.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    """
    comparison = _aligned_criterion_scores(results, "AICc", n_obs)
    return _mean_differences(comparison.model_names, comparison.scores, confidence)


def pairwise_cv_differences(
    results: Sequence[CVComparison], confidence: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare mean held-out NLL differences with t intervals.

    Parameters
    ----------
    results : Sequence[CVComparison]
        Named records with finite held-out NLLs in ``cv_fitness``. Participant
        IDs and pairing follow ``pairwise_aic_differences``; pooled fits are excluded.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.

    Returns
    -------
    means : pd.DataFrame
        Mean row-minus-column held-out NLL differences, without a parameter
        penalty. Negative favors the row model. Diagonals and comparisons with
        fewer than two participants are NaN.
    halfwidths : pd.DataFrame
        Two-sided Student-t interval half-widths, with matching labels and NaNs.

    """
    comparison = _align_model_scores(
        results,
        [model["cv_fitness"] for model in results],
    )
    return _mean_differences(comparison.model_names, comparison.scores, confidence)


def _winner_matrix(
    names: list[str], values: Float[np.ndarray, "models participants"]
) -> pd.DataFrame:
    """Tabulate strict-win fractions for aligned participant scores."""
    winners = np.mean(values[:, None, :] < values[None, :, :], axis=2)
    np.fill_diagonal(winners, np.nan)
    labels = pd.Index(names)
    return pd.DataFrame(winners, index=labels, columns=labels)


def winner_comparison_matrix(results: Sequence[ModelComparison]) -> pd.DataFrame:
    """Calculate participant-wise AIC strict-win fractions.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant fits following ``pairwise_aic_differences``.

    Returns
    -------
    pd.DataFrame
        Fraction of aligned participants with lower AIC in the row model than
        the column model. Ties count as wins for neither model; diagonals are NaN.

    """
    comparison = _aligned_criterion_scores(results)
    return _winner_matrix(comparison.model_names, comparison.scores)


def aicc_winner_comparison_matrix(
    results: Sequence[ModelComparison], n_obs: ArrayLike
) -> pd.DataFrame:
    """Calculate participant-wise AICc strict-win fractions.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant fits following ``pairwise_aic_differences``.
    n_obs : array_like
        Observation counts in the first record's original participant order,
        following ``calculate_aicc``.

    Returns
    -------
    pd.DataFrame
        Fraction of aligned participants with lower AICc in the row model than
        the column model. Ties count as wins for neither model; diagonals are NaN.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    """
    comparison = _aligned_criterion_scores(results, "AICc", n_obs)
    return _winner_matrix(comparison.model_names, comparison.scores)


def raw_winner_comparison_matrix(results: Sequence[ObjectiveSummary]) -> pd.DataFrame:
    """Calculate strict-win fractions for raw objective values.

    Parameters
    ----------
    results : Sequence[ObjectiveSummary]
        Named finite objectives, with lower values preferred. Participant IDs
        and pairing follow ``pairwise_aic_differences``; pooled fits are excluded.

    Returns
    -------
    pd.DataFrame
        Fraction of aligned participants with lower objective values in the row
        model, without a parameter penalty. Ties are wins for neither model;
        diagonals are NaN.

    """
    comparison = _align_model_scores(results, [model["fitness"] for model in results])
    return _winner_matrix(comparison.model_names, comparison.scores)


def cv_winner_comparison_matrix(results: Sequence[CVComparison]) -> pd.DataFrame:
    """Calculate strict-win fractions for held-out NLLs.

    Parameters
    ----------
    results : Sequence[CVComparison]
        Named finite held-out NLLs in ``cv_fitness``. Participant IDs and pairing
        follow ``pairwise_aic_differences``; pooled fits are excluded.

    Returns
    -------
    pd.DataFrame
        Fraction of aligned participants with lower held-out NLL in the row
        model, without a parameter penalty. Ties are wins for neither model;
        diagonals are NaN.

    """
    comparison = _align_model_scores(
        results,
        [model["cv_fitness"] for model in results],
    )
    return _winner_matrix(comparison.model_names, comparison.scores)


def generate_t_p_matrices(
    results: Sequence[ObjectiveSummary],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate paired t-statistics and one-sided p-values.

    Parameters
    ----------
    results : Sequence[ObjectiveSummary]
        Named finite raw objectives. Participant IDs and pairing follow
        ``pairwise_aic_differences``; pooled fits are excluded.

    Returns
    -------
    statistics : pd.DataFrame
        Paired t-statistics for row-model minus column-model objectives.
    probabilities : pd.DataFrame
        One-sided p-values for the alternative that the row model's mean objective
        is lower. Undefined entries in both tables are empty strings.

    Notes
    -----
    These are descriptive raw-objective comparisons without complexity penalties.

    """
    comparison = _align_model_scores(results, [model["fitness"] for model in results])
    labels = pd.Index(comparison.model_names)
    statistics = pd.DataFrame(np.nan, index=labels, columns=labels)
    probabilities = statistics.copy()
    if comparison.scores.shape[1] > 1:
        for row, column in itertools.permutations(range(len(labels)), 2):
            statistics.iat[row, column], probabilities.iat[row, column] = ttest_rel(
                comparison.scores[row], comparison.scores[column], alternative="less"
            )
    return statistics.replace(np.nan, ""), probabilities.replace(np.nan, "")


def _median_differences(
    names: list[str],
    values: Float[np.ndarray, "models participants"],
    confidence: float,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tabulate paired median differences and bootstrap interval endpoints."""
    labels = pd.Index(names)
    medians = pd.DataFrame(np.nan, index=labels, columns=labels)
    lower, upper = medians.copy(), medians.copy()
    if values.shape[1] < 2:
        return medians, lower, upper
    rng = np.random.default_rng(42)
    tail = (1 - confidence) / 2
    for row, column in itertools.combinations(range(len(names)), 2):
        differences = values[row] - values[column]
        bootstrap_medians = [
            np.median(rng.choice(differences, size=len(differences), replace=True))
            for _ in range(n_bootstrap)
        ]
        median_difference = np.median(differences)
        lower_endpoint, upper_endpoint = np.percentile(
            bootstrap_medians, [100 * tail, 100 * (1 - tail)]
        )
        medians.iat[row, column] = median_difference
        medians.iat[column, row] = -median_difference
        lower.iat[row, column], upper.iat[row, column] = lower_endpoint, upper_endpoint
        lower.iat[column, row], upper.iat[column, row] = -upper_endpoint, -lower_endpoint
    return medians, lower, upper


def pairwise_median_aic_differences(
    results: Sequence[ModelComparison],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare median AIC differences with percentile-bootstrap intervals.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant fits following ``pairwise_aic_differences``.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.
    n_bootstrap : int, optional
        Positive number of participant resamples per pair; default is 10,000.

    Returns
    -------
    medians : pd.DataFrame
        Median row-minus-column AIC differences; negative favors the row model.
    lower : pd.DataFrame
        Lower percentile-bootstrap interval endpoints.
    upper : pd.DataFrame
        Upper interval endpoints. All three tables have NaN diagonals and NaNs
        for comparisons with fewer than two participants.

    Notes
    -----
    Aligned participants are sampled with replacement using seed 42. Each unordered
    pair is resampled once. Reverse entries negate the median and swap and negate
    the interval endpoints. The interval need not be centered on the median.

    """
    comparison = _aligned_criterion_scores(results)
    return _median_differences(
        comparison.model_names, comparison.scores, confidence, n_bootstrap
    )


def pairwise_median_aicc_differences(
    results: Sequence[ModelComparison],
    n_obs: ArrayLike,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare median AICc differences with percentile-bootstrap intervals.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant fits following ``pairwise_aic_differences``.
    n_obs : array_like
        Observation counts in the first record's original participant order,
        following ``calculate_aicc``.
    confidence : float, optional
        Confidence level strictly between zero and one; default is 0.95.
    n_bootstrap : int, optional
        Positive number of participant resamples per pair; default is 10,000.

    Returns
    -------
    medians : pd.DataFrame
        Median row-minus-column AICc differences; negative favors the row model.
    lower : pd.DataFrame
        Lower percentile-bootstrap interval endpoints.
    upper : pd.DataFrame
        Upper interval endpoints. NaNs, paired resampling, seed 42 and reverse
        entries follow ``pairwise_median_aic_differences``.

    Raises
    ------
    ValueError
        If any observation count is less than or equal to its parameter count
        plus one.

    """
    comparison = _aligned_criterion_scores(results, "AICc", n_obs)
    return _median_differences(
        comparison.model_names, comparison.scores, confidence, n_bootstrap
    )


def floor_nested_fitness(
    results: Sequence[ModelComparison],
    nesting_pairs: Sequence[tuple[str, str] | list[str]],
) -> tuple[list[ModelComparison], pd.DataFrame]:
    """Create adjusted nested-model comparison scores and a change table.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named likelihood records with matching participant IDs, or pooled records
        using ``subject=[-1]``. Raw records remain unchanged.
    nesting_pairs : Sequence[tuple[str, str] or list[str]]
        Parent and child model names. Each pair applies the elementwise minimum
        NLL to the child, in supplied pair order. Stringified expressions are
        not parsed.

    Returns
    -------
    records : list[ModelComparison]
        Copies containing names, free-parameter names, adjusted scores, IDs and
        ``comparison_only=True``. Fitted parameter estimates are not included.
    changes : pd.DataFrame
        Columns subject, parent, child, original_nll and adjusted_nll. A pooled
        fit uses subject -1. Chained adjustments retain the original raw child
        score in each change record.

    Notes
    -----
    This reporting adjustment trusts declared nesting; it does not verify a
    feasible parameter mapping or produce an optimized fit. Use raw records for
    parameter summaries and simulation. Use the labelled comparison records only
    for adjusted in-sample comparisons, not cross-validation.

    """
    comparison = _align_model_scores(results, [model["fitness"] for model in results])
    adjusted_scores = comparison.scores.copy()
    positions = {name: row for row, name in enumerate(comparison.model_names)}
    changes = []
    for parent, child in nesting_pairs:
        parent_row, child_row = positions[parent], positions[child]
        parent_scores = adjusted_scores[parent_row]
        child_scores = adjusted_scores[child_row]
        improved = child_scores > parent_scores
        child_scores[improved] = parent_scores[improved]
        changes.extend(
            {
                "subject": comparison.subject_ids[row],
                "parent": parent,
                "child": child,
                "original_nll": comparison.scores[child_row, row],
                "adjusted_nll": child_scores[row],
            }
            for row in np.flatnonzero(improved)
        )
    records: list[ModelComparison] = []
    for model, scores in zip(results, adjusted_scores):
        record: ModelComparison = {
            "name": model["name"],
            "free": list(model["free"]),
            "fitness": scores.tolist(),
            "subject": comparison.subject_ids.copy(),
            "comparison_only": True,
        }
        records.append(record)
    return records, pd.DataFrame(
        changes,
        columns=pd.Index(
            ["subject", "parent", "child", "original_nll", "adjusted_nll"]
        ),
    )


def random_effects_model_selection(
    log_evidence: Float[Array | np.ndarray, "participants models"],
    model_names: Sequence[str],
    *,
    alpha0: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_samples: int = 100_000,
) -> pd.DataFrame:
    """Infer population model frequencies from participant log evidence.

    Parameters
    ----------
    log_evidence : Float[Array | np.ndarray, "participants models"]
        Nonempty finite matrix with the same participants in each model column
        and comparable observations. Higher values favor a model. The caller
        supplies integrated evidence or explicitly chooses an approximation.
    model_names : Sequence[str]
        Names labelling the model columns in their supplied order.
    alpha0 : float, optional
        Positive symmetric Dirichlet prior concentration per model; default is 1.
    max_iter : int, optional
        Positive variational-iteration limit; default is 1,000.
    tol : float, optional
        Positive tolerance on maximum absolute concentration change;
        default is 1e-8.
    n_samples : int, optional
        Positive number of Dirichlet draws for exceedance probabilities;
        default is 100,000, using seed 0.

    Returns
    -------
    pd.DataFrame
        Model, Expected Frequency, Exceedance Probability, Protected XP and Alpha
        columns, in supplied model order.

    Raises
    ------
    RuntimeError
        If the maximum absolute concentration change does not fall below ``tol``
        within ``max_iter`` iterations.

    Notes
    -----
    Inputs remain unchanged. This operation adds no information-criterion penalty.
    Pooled fits are not participant samples.

    References
    ----------
    .. [1] Stephan et al. (2009), NeuroImage 46:1004-1017.
    .. [2] Rigoux et al. (2014), NeuroImage 84:971-985, equations A.17/A.20
           and protected exceedance probabilities.

    Reference decomposition: https://github.com/spm/spm/blob/main/spm_BMS_bor.m

    """
    from scipy.special import digamma, expit, gammaln, logsumexp, softmax, xlogy

    log_evidence = np.array(log_evidence, dtype=float, copy=True)
    # Subtracting a participant's maximum improves stability without changing comparisons.
    log_evidence -= log_evidence.max(axis=1, keepdims=True)
    model_count = len(model_names)
    # Concentrations are the prior plus expected participant counts per model.
    posterior_concentration = np.full(model_count, alpha0, dtype=float)
    maximum_change = np.inf
    for _ in range(max_iter):
        participant_probabilities = softmax(
            log_evidence
            + digamma(posterior_concentration)
            - digamma(posterior_concentration.sum()),
            axis=1,
        )
        next_concentration = alpha0 + participant_probabilities.sum(axis=0)
        maximum_change = np.max(np.abs(next_concentration - posterior_concentration))
        posterior_concentration = next_concentration
        if maximum_change < tol:
            break
    else:
        raise RuntimeError(
            f"BMS did not converge after {max_iter} iterations (alpha change {maximum_change:g})"
        )
    # Recompute assignment probabilities with the final, converged concentrations.
    expected_log_frequency = digamma(posterior_concentration) - digamma(
        posterior_concentration.sum()
    )
    participant_probabilities = softmax(log_evidence + expected_log_frequency, axis=1)
    # The variational bound combines fit, uncertainty in participant assignments,
    # and the posterior/prior difference in population model frequencies (Rigoux A.20).
    expected_log_joint = np.sum(
        participant_probabilities * (log_evidence + expected_log_frequency)
    )
    assignment_entropy = -np.sum(
        xlogy(participant_probabilities, participant_probabilities)
    )
    dirichlet_contribution = (
        gammaln(model_count * alpha0)
        - model_count * gammaln(alpha0)
        + np.sum(gammaln(posterior_concentration))
        - gammaln(posterior_concentration.sum())
        + np.sum((alpha0 - posterior_concentration) * expected_log_frequency)
    )
    free_energy = expected_log_joint + assignment_entropy + dirichlet_contribution
    equal_frequency_log_evidence = np.sum(
        logsumexp(log_evidence, axis=1) - np.log(model_count)
    )
    equal_frequency_probability = expit(equal_frequency_log_evidence - free_energy)
    # Exceedance counts how often each model has the highest sampled frequency.
    frequency_draws = np.random.default_rng(0).dirichlet(
        posterior_concentration, size=n_samples
    )
    exceedance_probability = (
        np.bincount(frequency_draws.argmax(axis=1), minlength=model_count) / n_samples
    )
    return pd.DataFrame(
        {
            "Model": list(model_names),
            "Expected Frequency": posterior_concentration
            / posterior_concentration.sum(),
            "Exceedance Probability": exceedance_probability,
            "Protected XP": equal_frequency_probability / model_count
            + (1 - equal_frequency_probability) * exceedance_probability,
            "Alpha": posterior_concentration,
        }
    )


def bayesian_model_selection(
    results: Sequence[ModelComparison],
    alpha0: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_samples: int = 100_000,
    n_obs: ArrayLike | None = None,
) -> pd.DataFrame:
    """Infer model frequencies using the AIC/AICc evidence approximation.

    Parameters
    ----------
    results : Sequence[ModelComparison]
        Named participant likelihood fits following ``pairwise_aic_differences``.
        Scores must be finite and participant sets must match. Pooled fits are
        excluded.
    alpha0 : float, optional
        Positive symmetric Dirichlet prior concentration per model; default is 1.
    max_iter : int, optional
        Positive variational-iteration limit; default is 1,000.
    tol : float, optional
        Positive tolerance on maximum absolute concentration change;
        default is 1e-8.
    n_samples : int, optional
        Positive number of draws for exceedance probabilities;
        default is 100,000, using seed 0.
    n_obs : array_like or None, optional
        Observation counts in the first record's original participant order.
        None uses AIC; supplied counts use AICc and must exceed each parameter
        count plus one.

    Returns
    -------
    pd.DataFrame
        Model, Expected Frequency, Exceedance Probability, Protected XP and Alpha
        columns, in supplied model order.

    Raises
    ------
    ValueError
        If an AICc observation count is less than or equal to its parameter count
        plus one.
    RuntimeError
        If the variational iteration does not converge within ``max_iter``.

    Notes
    -----
    This interface aligns the records and approximates log evidence as ``-IC/2``;
    it does not compute an integrated model likelihood. Inference is performed by
    ``random_effects_model_selection``, which also accepts log evidence directly.

    """
    comparison = _aligned_criterion_scores(results, "AIC" if n_obs is None else "AICc", n_obs)
    return random_effects_model_selection(
        -0.5 * comparison.scores.T,
        comparison.model_names,
        alpha0=alpha0,
        max_iter=max_iter,
        tol=tol,
        n_samples=n_samples,
    )
