"""Parameter summary and model comparison utilities.

Provides functions for loading optimized parameters, computing
confidence intervals, generating t-test matrices, and computing
AIC/AICc/BIC model comparison statistics.

"""

import ast
import itertools
import json
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax.tree_util import tree_map
from scipy.stats import t, ttest_rel

from jaxcmr.typing import Array, Float, MemorySearchCreateFn

__all__ = [
    "add_summary_lines",
    "aicc_winner_comparison_matrix",
    "bayesian_model_selection",
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
    "load_opt_params",
    "pairwise_aic_differences",
    "pairwise_aicc_differences",
    "pairwise_cv_differences",
    "pairwise_median_aic_differences",
    "pairwise_median_aicc_differences",
    "raw_winner_comparison_matrix",
    "summarize_parameters",
    "validate_params",
    "winner_comparison_matrix",
]


def bound_params(
    params: Mapping[str, Float[Array, " popsize"]], bounds: Mapping[str, list[float]]
) -> dict[str, Float[Array, " popsize"]]:
    """Return parameters scaled within bounds"""
    return tree_map(
        lambda param, bound: jnp.minimum(jnp.maximum(bound[0], param), bound[1]),
        params,
        bounds,
    )


def load_opt_params(base_param_path: str):
    """Load the base parameters and bounds for the optimization.

    Args:
        base_param_path: Path to the base parameters file.
    """
    with open(base_param_path) as f:
        fit_config = json.load(f)
    base_params = fit_config["fixed"].copy()
    param_bounds = fit_config["free"].copy()
    return {"base": base_params, "bounds": param_bounds}


def validate_params(
    loss_fn, model_init: MemorySearchCreateFn, trials, list_arg, opt_params
):
    """Validate the bounds of the optimization parameters."""
    base_params, base_bounds = opt_params
    test_params = base_params.copy()
    for key, key_bounds in base_bounds.items():
        test_params[key] = key_bounds[0]
    loss_fn(model_init, list_arg, trials, test_params)
    for key, key_bounds in base_bounds.items():
        test_params[key] = key_bounds[1]
    loss_fn(model_init, list_arg, trials, test_params)


def calculate_ci(data: list[float], confidence=0.95) -> float:
    """Returns the confidence interval for a list of values.

    Args:
        data (list[float]): Values to calculate the confidence interval for.
        confidence (float, optional): The confidence level for the interval. Defaults to 0.95.
    """
    assert len(data) > 1
    n = len(data)
    stderr = np.std(np.array(data), ddof=1) / np.sqrt(n)
    return stderr * t.ppf((1 + confidence) / 2.0, n - 1)


def _normalize_variant(values: list[float]) -> np.ndarray | None:
    """Returns normalized variant values or None when no usable data is present.

    Args:
      values: Subject-level measurements for a single model variant.
    """
    variant_array = np.asarray(values, dtype=float)
    if variant_array.size == 0 or np.isnan(variant_array).all():
        return None
    return variant_array


def _format_row(parameter_label: str, statistic_label: str, values: list[str]) -> str:
    """Returns a Markdown table row for the summary output.

    Args:
      parameter_label: Name of the parameter or metric for the row.
      statistic_label: Statistic identifier (e.g., mean, std).
      values: Formatted statistic values for each model variant.
    """
    cells = [parameter_label, statistic_label, *values]
    sanitized = [cell or "" for cell in cells]
    return "| " + " | ".join(sanitized) + " |\n"


def _format_mean_cell(
    array: np.ndarray | None, raw_values: list[float], include_ci: bool
) -> str:
    """Returns the formatted mean cell text, optionally with confidence intervals.

    Args:
      array: Prepared numeric data for a model variant.
      raw_values: Original values for confidence interval calculation.
      include_ci: Whether to append the confidence interval.
    """
    if array is None:
        return ""
    mean_value = np.mean(array)
    if np.isnan(mean_value):
        return ""
    if include_ci:
        return f"{mean_value:.2f} +/- {calculate_ci(raw_values):.2f}"
    return f"{mean_value:.2f}"


def _format_stat_cell(
    array: np.ndarray | None, reducer: Callable[[np.ndarray], float]
) -> str:
    """Returns the formatted statistic cell using the provided reducer.

    Args:
      array: Prepared numeric data for a model variant.
      reducer: Function that computes the statistic of interest.
    """
    if array is None:
        return ""
    value = reducer(array)
    return "" if np.isnan(value) else f"{value:.2f}"


def add_summary_lines(
    md_table: str,
    errors: list[list[float]],
    label: str,
    include_std=False,
    include_ci=False,
) -> str:
    """Add summary statistics rows to a Markdown table segment.

    Args:
      md_table: Markdown table fragment to extend.
      errors: Values grouped by model variant.
      label: Parameter or metric name for the new rows.
      include_std: Whether to include standard deviation rows.
      include_ci: Whether to include confidence interval text for means.
    """
    display_label = label.replace("_", " ")
    variant_arrays = [_normalize_variant(values) for values in errors]

    mean_values = [
        _format_mean_cell(variant_array, raw_values, include_ci)
        for variant_array, raw_values in zip(variant_arrays, errors)
    ]
    md_table += _format_row(display_label, "mean", mean_values)

    if include_std:
        std_values = [
            _format_stat_cell(variant_array, np.std) for variant_array in variant_arrays
        ]
        md_table += _format_row("", "std", std_values)

    min_values = [
        _format_stat_cell(variant_array, np.nanmin) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "min", min_values)

    max_values = [
        _format_stat_cell(variant_array, np.nanmax) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "max", max_values)

    return md_table


def summarize_parameters(
    model_data: list[dict],
    query_parameters: list[str] | None = None,
    include_std=False,
    include_ci=False,
) -> str:
    """Returns a Markdown table of parameter statistics across model variants.

    The table includes the mean (with optional confidence intervals), standard deviation,
    minimum, and maximum for each requested parameter.

    Args:
      model_data: Collection of model summaries containing `name`, `fitness`, and `fits`.
      query_parameters: Ordered parameter identifiers to surface; defaults to all found.
      include_std: Whether to include standard deviation rows.
      include_ci: Whether to append confidence intervals to mean rows.
    """
    # Extract model names in input order
    model_names = [variant["name"] for variant in model_data]

    # identify query parameters; by default, is all unique fixed params across model variants
    if query_parameters is None:
        query_parameters = []
        for entry in model_data:
            for param in entry["fixed"]:
                if param not in query_parameters:
                    query_parameters.append(param)

    header_names = [n.replace("_", " ") for n in model_names]
    md_table = "| Parameter | Statistic | " + " | ".join(header_names) + " |\n"
    md_table += "|---|---" + ("|---" * len(model_data)) + "|\n"

    # likelihood entry first
    values = [variant_data["fitness"] for variant_data in model_data]
    md_table = add_summary_lines(
        md_table, values, "fitness", include_std=include_std, include_ci=include_ci
    )

    # Compute the mean and confidence interval for params for each model variant
    for param in query_parameters:
        values = []
        for variant_data in model_data:
            subject_count = len(variant_data["fitness"])
            fallback = [np.nan] * subject_count
            values.append(variant_data["fits"].get(param, fallback))
        md_table = add_summary_lines(
            md_table, values, param, include_std=include_std, include_ci=include_ci
        )

    return md_table


def _score_array(model: dict, field: str) -> np.ndarray:
    values = np.asarray(model[field], dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"{model['name']}: {field} must be a nonempty, finite 1D array"
        )
    return values


def _subject_ids(model: dict, size: int) -> list | None:
    """Read current fit IDs, legacy IDs, or CV IDs without changing their order."""
    candidates = [
        source[key]
        for source, key in [
            (model.get("fits", {}), "subject"),
            (model, "subject"),
            (model, "subjects"),
        ]
        if key in source
    ]
    identifiers = None
    for candidate in candidates:
        array = np.asarray(candidate, dtype=object)
        if array.ndim != 1 or len(array) != size:
            raise ValueError(
                f"{model['name']}: subject identifiers must match the score rows"
            )
        values = array.tolist()
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, (str, int, np.integer))
            for x in values
        ):
            raise ValueError(
                f"{model['name']}: subject identifiers must be integers or strings"
            )
        if len(set(values)) != size:
            raise ValueError(f"{model['name']}: duplicate subject identifiers")
        if identifiers is not None and identifiers != values:
            raise ValueError(f"{model['name']}: conflicting subject identifier fields")
        identifiers = values
    return identifiers


def _comparison_data(
    results: list[dict], field: str = "fitness", *, participantwise: bool = False
) -> tuple[list[str], np.ndarray, list | None]:
    """Align to a stable ID order; an unidentified single pooled row is allowed for ICs."""
    if not results:
        raise ValueError("At least one model result is required")
    names = [model["name"] for model in results]
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Model names must be nonempty and unique")
    arrays = [_score_array(model, field) for model in results]
    ids = [_subject_ids(model, len(values)) for model, values in zip(results, arrays)]
    if any(len(values) != len(arrays[0]) for values in arrays):
        raise ValueError("Models must contain the same score rows and subject set")
    if any(subjects is None for subjects in ids):
        if (
            participantwise
            or len(arrays[0]) > 1
            or any(subjects is not None for subjects in ids)
        ):
            raise ValueError(
                "Subject identifiers are required for participant-wise comparisons"
            )
        return names, np.stack(arrays), None
    if any(set(subjects) != set(ids[0]) for subjects in ids):
        raise ValueError("Models must contain the same subject set")
    if participantwise and ids[0] == [-1]:
        raise ValueError("A pooled fit is not a participant-wise sample")
    order = sorted(ids[0], key=lambda value: (isinstance(value, str), value))
    aligned = []
    for subjects, values in zip(ids, arrays):
        positions = {subject: i for i, subject in enumerate(subjects)}
        aligned.append(values[[positions[subject] for subject in order]])
    return names, np.stack(aligned), order


def _free_parameters(model: dict) -> list[str]:
    free = model["free"]
    if not isinstance(free, (Mapping, list, tuple)):
        raise ValueError(f"{model['name']}: free must list the fitted parameter names")
    names = list(free)
    if any(not isinstance(name, str) or not name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError(
            f"{model['name']}: free parameter names must be unique strings"
        )
    return names


def _observation_counts(n_obs, size: int) -> np.ndarray:
    counts = np.asarray(n_obs, dtype=float)
    if counts.ndim == 0 and size == 1:
        counts = counts.reshape(1)
    if (
        counts.ndim != 1
        or len(counts) != size
        or not np.isfinite(counts).all()
        or np.any(counts <= 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError(
            "n_obs must contain one positive integer per fit row; scalars require one row"
        )
    return counts


def _subjectwise_aic(model: dict) -> np.ndarray:
    """Per-fit AIC in original row order; retained for existing research callers."""
    return 2 * _score_array(model, "fitness") + 2 * len(_free_parameters(model))


def _subjectwise_aicc(model: dict, n_obs: np.ndarray) -> np.ndarray:
    """Per-fit AICc in original row order, with explicitly matching counts."""
    aic = _subjectwise_aic(model)
    counts = _observation_counts(n_obs, len(aic))
    k = len(_free_parameters(model))
    if np.any(counts <= k + 1):
        raise ValueError("AICc undefined where n_obs <= k + 1")
    return aic + 2 * k * (k + 1) / (counts - k - 1)


def _criterion_data(
    results: list[dict],
    criterion: str = "AIC",
    n_obs=None,
    *,
    participantwise: bool = False,
) -> tuple[list[str], np.ndarray, list | None]:
    names, nll, ids = _comparison_data(results, participantwise=participantwise)
    k = np.array([len(_free_parameters(model)) for model in results])[:, None]
    if criterion == "AIC":
        return names, 2 * nll + 2 * k, ids
    counts = _observation_counts(n_obs, nll.shape[1])
    # Count vectors refer to the FIRST record's original row order, before alignment.
    if ids is not None:
        original = _subject_ids(results[0], len(counts))
        positions = {subject: i for i, subject in enumerate(original)}
        counts = counts[[positions[subject] for subject in ids]]
    if criterion == "BIC":
        return names, 2 * nll + k * np.log(counts), ids
    denominator = counts - k - 1
    if np.any(denominator <= 0):
        raise ValueError("AICc undefined where n_obs <= k + 1")
    return names, 2 * nll + 2 * k + 2 * k * (k + 1) / denominator, ids


def _criterion_table(results: list[dict], criterion: str, n_obs=None) -> pd.DataFrame:
    names, values, _ = _criterion_data(results, criterion, n_obs)
    return pd.DataFrame({"Model": names, criterion: values.sum(axis=1)}).sort_values(
        criterion
    )


def calculate_aic(results: list[dict]) -> pd.DataFrame:
    """Return AIC = sum_rows(2*NLL + 2*k), lowest first.

    Each record supplies ``name``, ``free`` parameter names, and a 1D ``fitness``
    array of SUMMED negative log likelihoods, one per independently fitted
    parameter set. A pooled fit has one row and incurs one penalty. Multiple
    rows require matching subject identifiers (see ``pairwise_aic_differences``).
    This convention also applies to AICc, BIC and information-criterion weights.
    Generic optimizer objectives such as MSE or mean/scaled NLL must not be used.
    Neither the objective nor its sign can be inferred from its numerical value.
    """
    return _criterion_table(results, "AIC")


def calculate_bic_scores(results: list[dict], n_obs: np.ndarray | int) -> pd.DataFrame:
    """Return sum_rows(2*NLL + k*log(n_obs)), lowest first.

    ``n_obs`` gives positive integer observation counts in the FIRST record's
    row order; it is realigned with the subject IDs. A scalar is allowed only
    for one fitted row. The caller defines the scientific observation unit;
    the number of fitted participants is never substituted for these counts.
    Scores and parameter counts follow ``calculate_aic``.
    """
    return _criterion_table(results, "BIC", n_obs)


def calculate_aicc(results: list[dict], n_obs: np.ndarray | int) -> pd.DataFrame:
    """Return sum_rows(AIC + 2*k*(k+1)/(n_obs-k-1)), lowest first.

    The caller supplies observation counts in the first record's row order,
    as for ``calculate_bic_scores``, and determines whether this finite-sample
    correction is appropriate for the likelihood. Raise ValueError when any
    count is <= k+1. A scalar count is allowed only for one fitted row.
    """
    return _criterion_table(results, "AICc", n_obs)


def _criterion_weights(
    table: pd.DataFrame, criterion: str, column: str
) -> pd.DataFrame:
    delta = table[criterion] - table[criterion].min()
    weights = np.exp(-0.5 * delta)
    return (
        table[["Model"]]
        .assign(**{column: weights / weights.sum()})
        .sort_values(column, ascending=False)
    )


def calculate_aic_weights(results: list[dict]) -> pd.DataFrame:
    """Return weights from ``calculate_aic`` scores, largest first (Model, AICw)."""
    return _criterion_weights(calculate_aic(results), "AIC", "AICw")


def calculate_aicc_weights(
    results: list[dict], n_obs: np.ndarray | int
) -> pd.DataFrame:
    """Return weights from ``calculate_aicc`` scores, largest first (Model, AICcw)."""
    return _criterion_weights(calculate_aicc(results, n_obs), "AICc", "AICcw")


def _validate_confidence(confidence: float) -> None:
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def _mean_differences(names: list[str], values: np.ndarray, confidence: float):
    _validate_confidence(confidence)
    mean = np.full((len(names), len(names)), np.nan)
    halfwidth = mean.copy()
    if values.shape[1] > 1:
        for i, j in itertools.product(range(len(names)), repeat=2):
            if i != j:
                difference = values[i] - values[j]
                mean[i, j] = np.mean(difference)
                halfwidth[i, j] = calculate_ci(difference.tolist(), confidence)
    return tuple(
        pd.DataFrame(array, index=names, columns=names) for array in (mean, halfwidth)
    )


def pairwise_aic_differences(
    results: list[dict], confidence: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean participant-wise AIC differences and t-interval half-widths.

    Entries are row model minus column model: negative favors the row model.
    IDs come from ``fits['subject']``, legacy ``subject``, or CV ``subjects``;
    conflicting fields, duplicate IDs, unequal sets and nonfinite scores fail.
    Fit rows are aligned by ID. Pooled sentinel -1 is not a participant sample.
    Diagonals and comparisons with fewer than two participants are NaN.
    No Boolean equivalence decision is made. Scores follow ``calculate_aic``.
    """
    names, values, _ = _criterion_data(results, participantwise=True)
    return _mean_differences(names, values, confidence)


def pairwise_aicc_differences(
    results: list[dict], n_obs: np.ndarray, confidence: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean AICc differences and t-interval half-widths.

    Pairing and output follow ``pairwise_aic_differences``; counts follow
    ``calculate_aicc`` and refer to the first record's original row order.
    """
    names, values, _ = _criterion_data(results, "AICc", n_obs, participantwise=True)
    return _mean_differences(names, values, confidence)


def pairwise_cv_differences(
    results: list[dict], confidence: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean differences and t-interval half-widths for aligned ``cv_fitness``.

    Values are held-out NLL; negative row-minus-column differences favor the
    row model. Pairing and missing-value rules follow ``pairwise_aic_differences``.
    """
    names, values, _ = _comparison_data(results, "cv_fitness", participantwise=True)
    return _mean_differences(names, values, confidence)


def _winner_matrix(names: list[str], values: np.ndarray) -> pd.DataFrame:
    winners = np.mean(values[:, None, :] < values[None, :, :], axis=2)
    np.fill_diagonal(winners, np.nan)
    return pd.DataFrame(winners, index=names, columns=names)


def winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Fraction of aligned participants with strictly lower AIC in the row model.

    Ties are wins for neither model; diagonals are NaN. Identifiers and finite
    scores are required as for ``pairwise_aic_differences``.
    """
    names, values, _ = _criterion_data(results, participantwise=True)
    return _winner_matrix(names, values)


def aicc_winner_comparison_matrix(
    results: list[dict], n_obs: np.ndarray
) -> pd.DataFrame:
    """AICc strict-win fractions; counts follow ``calculate_aicc``, pairing follows AIC."""
    names, values, _ = _criterion_data(results, "AICc", n_obs, participantwise=True)
    return _winner_matrix(names, values)


def raw_winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Aligned strict-win fractions for raw objectives (lower is better), without a penalty."""
    names, values, _ = _comparison_data(results, participantwise=True)
    return _winner_matrix(names, values)


def cv_winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Aligned strict-win fractions for held-out ``cv_fitness`` (NLL), without a penalty."""
    names, values, _ = _comparison_data(results, "cv_fitness", participantwise=True)
    return _winner_matrix(names, values)


def generate_t_p_matrices(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired t-statistics and one-sided p-values for aligned raw objectives.

    The alternative is row < column. These are descriptive raw-fitness
    comparisons, without complexity penalties. Undefined cells remain empty
    strings for existing table formatting. Pairing follows the AIC helpers.
    """
    names, values, _ = _comparison_data(results, participantwise=True)
    statistics = np.full((len(names), len(names)), np.nan)
    probabilities = statistics.copy()
    if values.shape[1] > 1:
        for i, j in itertools.product(range(len(names)), repeat=2):
            if i != j:
                statistics[i, j], probabilities[i, j] = ttest_rel(
                    values[i], values[j], alternative="less"
                )
    return tuple(
        pd.DataFrame(array, index=names, columns=names).replace(np.nan, "")
        for array in (statistics, probabilities)
    )


def _median_differences(
    names: list[str], values: np.ndarray, confidence: float, n_bootstrap: int
):
    _validate_confidence(confidence)
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, (int, np.integer))
        or n_bootstrap <= 0
    ):
        raise ValueError("n_bootstrap must be a positive integer")
    median = np.full((len(names), len(names)), np.nan)
    lower, upper = median.copy(), median.copy()
    rng = np.random.default_rng(42)
    tail = (1 - confidence) / 2
    if values.shape[1] > 1:
        for i, j in itertools.product(range(len(names)), repeat=2):
            if i == j:
                continue
            difference = values[i] - values[j]
            median[i, j] = np.median(difference)
            samples = [
                np.median(rng.choice(difference, size=len(difference), replace=True))
                for _ in range(n_bootstrap)
            ]
            lower[i, j], upper[i, j] = np.percentile(
                samples, [100 * tail, 100 * (1 - tail)]
            )
    return tuple(
        pd.DataFrame(array, index=names, columns=names)
        for array in (median, lower, upper)
    )


def pairwise_median_aic_differences(
    results: list[dict], confidence: float = 0.95, n_bootstrap: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return median AIC differences and percentile-bootstrap lower/upper endpoints.

    Resample aligned participants with replacement (seed 42). The percentile
    interval need not be centered on the sample median. Pairing, direction
    and undefined cells follow ``pairwise_aic_differences``.
    """
    names, values, _ = _criterion_data(results, participantwise=True)
    return _median_differences(names, values, confidence, n_bootstrap)


def pairwise_median_aicc_differences(
    results: list[dict],
    n_obs: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median AICc differences with percentile-bootstrap endpoints (seed 42).

    Counts follow ``calculate_aicc``; output follows ``pairwise_median_aic_differences``.
    """
    names, values, _ = _criterion_data(results, "AICc", n_obs, participantwise=True)
    return _median_differences(names, values, confidence, n_bootstrap)


def floor_nested_fitness(
    results: list[dict], nesting_pairs: list[tuple[str, str]]
) -> tuple[list[dict], pd.DataFrame]:
    """Return separate nested-model comparison scores and a substitution table.

    Each (parent, child) declaration applies the elementwise minimum NLL in
    the supplied order, using matching IDs. Tuple, list and stringified pairs
    from notebook parameterization are accepted. Raw records remain untouched.
    Returned records contain only names, free parameter names, copied scores,
    IDs and a comparison-only marker, NEVER fitted parameter estimates.

    The table records subject, parent, child, original_nll and adjusted_nll
    for each change (subject is None for unidentified single-row pooled fits).
    Chained adjustments record the original raw child score at every step.
    This reporting adjustment trusts declared nesting; it neither verifies a
    feasible parameter mapping nor produces an optimized fit. Use raw records
    for parameter summaries and simulation. Use these explicitly labelled
    comparison records only for the adjusted in-sample comparisons, not CV.
    """
    names, raw, ids = _comparison_data(results)
    free = [_free_parameters(model) for model in results]
    values = raw.copy()
    positions = {name: i for i, name in enumerate(names)}
    changes = []
    for pair in nesting_pairs:
        if isinstance(pair, str):
            try:
                pair = ast.literal_eval(pair)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid nesting pair: {pair!r}") from exc
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(not isinstance(x, str) for x in pair)
        ):
            raise ValueError(f"Invalid nesting pair: {pair!r}")
        parent, child = pair
        if parent not in positions or child not in positions:
            raise ValueError(f"Unknown model in nesting pair: {pair!r}")
        p, c = positions[parent], positions[child]
        for row in np.flatnonzero(values[c] > values[p]):
            values[c, row] = values[p, row]
            changes.append(
                dict(
                    subject=None if ids is None else ids[row],
                    parent=parent,
                    child=child,
                    original_nll=raw[c, row],
                    adjusted_nll=values[c, row],
                )
            )
    records = []
    for name, params, scores in zip(names, free, values):
        record = dict(
            name=name, free=params, fitness=scores.tolist(), comparison_only=True
        )
        if ids is not None:
            record["subject"] = ids.copy()
        records.append(record)
    return records, pd.DataFrame(
        changes, columns=["subject", "parent", "child", "original_nll", "adjusted_nll"]
    )


def bayesian_model_selection(
    results: list[dict],
    alpha0: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_samples: int = 100_000,
    n_obs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Random-effects BMS using approximate per-participant log evidence -IC/2.

    IC is AIC, or AICc when observation counts are supplied in the first
    record's row order. This is an evidence approximation, not an integrated
    model likelihood. Participant IDs and finite scores are required. Pooled
    fits are not participant samples. alpha0 is the positive symmetric
    Dirichlet prior per model. Monte Carlo exceedance uses seed 0.

    Returns Model, Expected Frequency, Exceedance Probability, Protected XP,
    Alpha. Raises RuntimeError if the variational iteration fails to converge
    within max_iter (maximum absolute alpha change < tol).

    References: Stephan et al. (2009), NeuroImage 46:1004–1017; Rigoux et al.
    (2014), NeuroImage 84:971–985, equations A.17/A.20 and protected XP.
    Reference decomposition: https://github.com/spm/spm/blob/main/spm_BMS_bor.m
    """
    from scipy.special import digamma, expit, gammaln, logsumexp, softmax, xlogy

    names, criteria, _ = _criterion_data(
        results, "AIC" if n_obs is None else "AICc", n_obs, participantwise=True
    )
    if len(names) < 2:
        raise ValueError("BMS requires at least two models")
    if not np.isfinite(alpha0) or alpha0 <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("alpha0 and tol must be finite and positive")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0
        for x in (max_iter, n_samples)
    ):
        raise ValueError("max_iter and n_samples must be positive integers")
    evidence = -0.5 * criteria.T
    evidence -= evidence.max(axis=1, keepdims=True)  # harmless per-subject offsets
    k = len(names)
    alpha = np.full(k, alpha0, dtype=float)
    for _ in range(max_iter):
        responsibility = softmax(
            evidence + digamma(alpha) - digamma(alpha.sum()), axis=1
        )
        updated = alpha0 + responsibility.sum(axis=0)
        change = np.max(np.abs(updated - alpha))
        alpha = updated
        if change < tol:
            break
    else:
        raise RuntimeError(
            f"BMS did not converge after {max_iter} iterations (alpha change {change:g})"
        )
    expected_log_frequency = digamma(alpha) - digamma(alpha.sum())
    responsibility = softmax(evidence + expected_log_frequency, axis=1)
    # Expected log joint plus categorical and Dirichlet entropies (Rigoux A.20).
    free_energy = (
        np.sum(responsibility * (evidence + expected_log_frequency))
        - np.sum(xlogy(responsibility, responsibility))
        + gammaln(k * alpha0)
        - k * gammaln(alpha0)
        + np.sum(gammaln(alpha))
        - gammaln(alpha.sum())
        + np.sum((alpha0 - alpha) * expected_log_frequency)
    )
    null_evidence = np.sum(logsumexp(evidence, axis=1) - np.log(k))
    omnibus_risk = expit(null_evidence - free_energy)
    draws = np.random.default_rng(0).dirichlet(alpha, size=n_samples)
    xp = np.bincount(draws.argmax(axis=1), minlength=k) / n_samples
    return pd.DataFrame(
        {
            "Model": names,
            "Expected Frequency": alpha / alpha.sum(),
            "Exceedance Probability": xp,
            "Protected XP": omnibus_risk / k + (1 - omnibus_risk) * xp,
            "Alpha": alpha,
        }
    )
