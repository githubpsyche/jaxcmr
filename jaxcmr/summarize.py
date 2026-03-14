"""Parameter summary and model comparison utilities.

Provides functions for loading optimized parameters, computing
confidence intervals, generating t-test matrices, and computing
AIC/BIC model comparison statistics.

"""

import json
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax.tree_util import tree_map
from scipy.stats import t, ttest_rel

from jaxcmr.typing import Array, Float, MemorySearchCreateFn

__all__ = [
    "bound_params",
    "load_opt_params",
    "validate_params",
    "calculate_ci",
    "add_summary_lines",
    "summarize_parameters",
    "generate_t_p_matrices",
    "calculate_aic_weights",
    "calculate_aic",
    "calculate_bic_scores",
    "pairwise_aic_differences",
    "winner_comparison_matrix",
    "raw_winner_comparison_matrix",
    "pairwise_cv_differences",
    "cv_winner_comparison_matrix",
    "pairwise_median_aic_differences",
    "bayesian_model_selection",
    "floor_nested_fitness",
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


def _normalize_variant(values: list[float]) -> Optional[np.ndarray]:
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
    array: Optional[np.ndarray], raw_values: list[float], include_ci: bool
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
    array: Optional[np.ndarray], reducer: Callable[[np.ndarray], float]
) -> str:
    """Returns the formatted statistic cell using the provided reducer.

    Args:
      array: Prepared numeric data for a model variant.
      reducer: Function that computes the statistic of interest.
    """
    if array is None:
        return ""
    value = reducer(array)
    if np.isnan(value):
        return ""
    return f"{value:.2f}"


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
    query_parameters: Optional[list[str]] = None,
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
            for param in entry["fixed"].keys():
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


def generate_t_p_matrices(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns matrices of t-values and p-values from paired t-tests on model fitness results.

    Args:
    - results: dicts containing `name` and subjectwise `fitness` data for each model.
    """
    # Extract model names
    model_names = [model["name"] for model in results]
    num_models = len(model_names)

    # Initialize matrices for t-values and 'less' p-values
    t_values = np.zeros((num_models, num_models))
    p_values_less = np.zeros((num_models, num_models))

    # Populate the matrices with t and p values from the paired t-tests
    for i, model_a in enumerate(results):
        for j, model_b in enumerate(results):
            t, p_less = ttest_rel(
                model_a["fitness"], model_b["fitness"], alternative="less"
            )
            t_values[i, j] = t
            p_values_less[i, j] = p_less

    # Convert matrices to pandas DataFrames, replacing NaNs with empty strings
    df_t = pd.DataFrame(t_values, index=model_names, columns=model_names).replace(
        np.nan, ""
    )
    df_p = pd.DataFrame(p_values_less, index=model_names, columns=model_names).replace(
        np.nan, ""
    )

    return df_t, df_p


def calculate_aic_weights(results: list[dict]) -> pd.DataFrame:
    """
    Calculates the Akaike Information Criterion weights for a list of models.

    Parameters:
    - results (list): A list of dictionaries, each with 'name', 'fitness', and 'free' (parameters).

    Returns:
    - DataFrame: A pandas DataFrame with a row for each model and their AICw scores.
    """
    aics = []
    names = []

    # Calculate AIC for each model
    for model in results:
        k = len(model["free"])  # number of parameters
        n_subjects = len(model["fitness"])
        total_nll = sum(model["fitness"])
        aic = 2 * k * n_subjects + 2 * total_nll
        aics.append(aic)
        names.append(model["name"])

    # Convert AICs to AIC weights
    aics = np.array(aics)
    min_aic = np.min(aics)
    delta_aic = aics - min_aic
    weights = np.exp(-0.5 * delta_aic)
    aic_weights = weights / np.sum(weights)

    # Create DataFrame
    df = pd.DataFrame({"Model": names, "AICw": aic_weights})
    return df.sort_values(by="AICw", ascending=False)

def calculate_aic(results: list[dict]) -> pd.DataFrame:
    """
    Return a DataFrame of Akaike Information Criterion (AIC) scores
    for a collection of model-fit summaries.

    Each `model` dictionary must contain:
      • "name"   – a string identifying the model
      • "fitness" – an iterable of per-observation log-likelihoods
      • "free"    – a collection of free parameters (length = k)

    AIC = 2k – 2·LL, where LL is the total log-likelihood.
    Lower AIC indicates a better trade-off between fit and complexity.
    """
    aics, names = [], []

    for model in results:
        k  = len(model["free"])          # number of free parameters
        ll = np.sum(model["fitness"])    # total log-likelihood (positive)
        aic = 2 * k - 2 * ll             # <-- note the minus sign
        aics.append(aic)
        names.append(model["name"])

    df = pd.DataFrame({"Model": names, "AIC": aics})
    return df.sort_values(by="AIC", ascending=True)  # best (lowest) first

def calculate_bic_scores(
    results: list[dict],
) -> pd.DataFrame:
    """
    Return a DataFrame of Bayesian Information Criterion (BIC) scores
    for a collection of model-fit summaries.
    """
    names, bics = [], []

    for model in results:
        k = len(model["free"])  # number of free parameters
        ll = np.sum(model["fitness"])  # total log-likelihood

        # determine sample size for this model
        n = len(model["fitness"])

        bic = k * np.log(n) - 2 * ll
        names.append(model["name"])
        bics.append(bic)

    df = pd.DataFrame({"Model": names, "BIC": bics})
    return df.sort_values(by="BIC", ascending=False)


def pairwise_aic_differences(
    results: list[dict],
    confidence: float = 0.95,
    equivalence_margin: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute pairwise AIC differences between models.

    Args:
      results: Model summaries containing `name`, per-subject `fitness`, and `free`.
      confidence: Confidence level for the ΔAIC intervals.
      equivalence_margin: Half-width in AIC units used for practical equivalence.

    Returns:
      (mean_delta, ci_halfwidth, equivalent): DataFrames with model names on rows
      and columns. For each pair (i, j):

      * mean_delta[i, j] is the mean ΔAIC_s = AIC_i,s − AIC_j,s across subjects.
        Negative values favor model i; positive values favor model j.
      * ci_halfwidth[i, j] is the half-width of the confidence interval for ΔAIC.
      * equivalent[i, j] is True when |mean_delta| ≤ equivalence_margin and the
        confidence interval includes zero, indicating practical equivalence.
    """
    model_names = [model["name"] for model in results]
    num_models = len(model_names)

    aic_values = [_subjectwise_aic(model) for model in results]

    mean_delta = np.full((num_models, num_models), np.nan, dtype=float)
    ci_halfwidth = np.full((num_models, num_models), np.nan, dtype=float)
    equivalent = np.zeros((num_models, num_models), dtype=bool)

    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue

            diff = aic_values[i] - aic_values[j]
            valid = ~np.isnan(diff)
            if np.count_nonzero(valid) <= 1:
                continue

            diff_valid = diff[valid]
            mean_value = float(np.mean(diff_valid))
            ci = calculate_ci(diff_valid.tolist(), confidence=confidence)

            mean_delta[i, j] = mean_value
            ci_halfwidth[i, j] = ci

            lower, upper = mean_value - ci, mean_value + ci
            equivalent[i, j] = (abs(mean_value) <= equivalence_margin) and (
                lower <= 0.0 <= upper
            )

    df_mean = pd.DataFrame(mean_delta, index=model_names, columns=model_names)
    df_ci = pd.DataFrame(ci_halfwidth, index=model_names, columns=model_names)
    df_equiv = pd.DataFrame(equivalent, index=model_names, columns=model_names)

    return df_mean, df_ci, df_equiv


def winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Returns matrix of fractions of AIC in model i < model j.

    Args:
        - results: dicts containing each containing 'name', 'fitness', and 'free' data.
            Per-subject AIC is computed as 2k + 2*NLL_s.
    """
    model_names = [model["name"] for model in results]
    num_models = len(model_names)
    penalized = [_subjectwise_aic(model) for model in results]

    comparison_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                comparison_matrix[i, j] = float(np.mean(penalized[i] < penalized[j]))
            else:
                comparison_matrix[i, j] = np.nan

    return pd.DataFrame(comparison_matrix, index=model_names, columns=model_names)


def raw_winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Returns matrix of fractions of fitness in row model < in model j.

    Args:
        - results: dicts containing each containing 'name' and 'fitness' data for each model.
    """
    # Extract model names
    model_names = [model["name"] for model in results]
    num_models = len(model_names)

    # Initialize matrix for comparison results
    comparison_matrix = np.zeros((num_models, num_models))

    # Populate the matrix with comparison fractions
    for i, model_a in enumerate(results):
        for j, model_b in enumerate(results):
            if i != j:
                # Calculate the fraction of fitness values in model_a that are lower than those in model_b
                comparison_scores = np.array(model_a["fitness"]) < np.array(
                    model_b["fitness"]
                )
                comparison_fraction = np.mean(comparison_scores)
                comparison_matrix[i, j] = comparison_fraction
            else:
                # Set diagonal to NaN for clarity, since self-comparison does not make sense here
                comparison_matrix[i, j] = np.nan

    return pd.DataFrame(comparison_matrix, index=model_names, columns=model_names)


def pairwise_cv_differences(
    results: list[dict],
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise CV-NLL differences between models.

    Parameters
    ----------
    results : list[dict]
        Model summaries with ``'name'`` and ``'cv_fitness'`` (per-subject CV-NLL).
    confidence : float
        Confidence level for the difference intervals.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (mean_delta, ci_halfwidth). Negative mean_delta[i, j] favors model i.

    """
    model_names = [m["name"] for m in results]
    n = len(model_names)
    cv_values = [np.asarray(m["cv_fitness"], dtype=float) for m in results]

    mean_delta = np.full((n, n), np.nan, dtype=float)
    ci_hw = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = cv_values[i] - cv_values[j]
            valid = ~np.isnan(diff)
            if np.count_nonzero(valid) <= 1:
                continue
            diff_valid = diff[valid]
            mean_delta[i, j] = float(np.mean(diff_valid))
            ci_hw[i, j] = calculate_ci(diff_valid.tolist(), confidence=confidence)

    return (
        pd.DataFrame(mean_delta, index=model_names, columns=model_names),
        pd.DataFrame(ci_hw, index=model_names, columns=model_names),
    )


def cv_winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
    """Return fraction of subjects where model i has lower CV-NLL than model j.

    Parameters
    ----------
    results : list[dict]
        Model summaries with ``'name'`` and ``'cv_fitness'``.

    Returns
    -------
    pd.DataFrame
        Matrix with model names as row/column labels.

    """
    model_names = [m["name"] for m in results]
    n = len(model_names)
    cv_values = [np.asarray(m["cv_fitness"], dtype=float) for m in results]

    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = float(np.mean(cv_values[i] < cv_values[j]))

    return pd.DataFrame(matrix, index=model_names, columns=model_names)


def _subjectwise_aic(model: dict) -> np.ndarray:
    """Return per-subject AIC values: AIC_s = 2k + 2*NLL_s.

    Parameters
    ----------
    model : dict
        Model summary with ``'fitness'`` (per-subject NLL) and ``'free'``.

    Returns
    -------
    np.ndarray
        Per-subject AIC values.

    """
    fitness = np.asarray(model["fitness"], dtype=float)
    penalty = 2.0 * len(model.get("free", []))
    return 2.0 * fitness + penalty


def floor_nested_fitness(
    results: list[dict],
    nesting_pairs: list[tuple[str, str]],
) -> int:
    """Floor child model fitness at parent fitness for nested model pairs.

    Parameters
    ----------
    results : list[dict]
        Model summaries with ``'name'`` and ``'fitness'``. Modified in place.
    nesting_pairs : list[tuple[str, str]]
        Each tuple is ``(parent_name, child_name)`` where the child nests
        the parent. The child's per-subject fitness is floored at the
        parent's value (element-wise minimum), correcting optimizer
        failures where a nested model reports worse NLL than its parent.

    Returns
    -------
    int
        Total number of subject-model entries that were floored.

    """
    name_to_result = {r["name"]: r for r in results}
    total_floored = 0

    for parent_name, child_name in nesting_pairs:
        if parent_name not in name_to_result:
            raise ValueError(
                f"Parent model '{parent_name}' not found in results. "
                f"Available: {list(name_to_result)}"
            )
        if child_name not in name_to_result:
            raise ValueError(
                f"Child model '{child_name}' not found in results. "
                f"Available: {list(name_to_result)}"
            )

        parent = name_to_result[parent_name]
        child = name_to_result[child_name]
        parent_fitness = np.asarray(parent["fitness"], dtype=float)
        child_fitness = np.asarray(child["fitness"], dtype=float)
        n_floored = int(np.sum(child_fitness > parent_fitness))
        total_floored += n_floored
        child["fitness"] = np.minimum(child_fitness, parent_fitness).tolist()

    return total_floored


def pairwise_median_aic_differences(
    results: list[dict],
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise median AIC differences between models.

    Parameters
    ----------
    results : list[dict]
        Model summaries with ``'name'``, ``'fitness'``, and ``'free'``.
    confidence : float
        Confidence level for the bootstrap interval.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (median_delta, ci_halfwidth). For each pair (i, j),
        median_delta[i, j] is the median of AIC_i,s - AIC_j,s.
        Negative values favor model i; positive values favor model j.

    """
    model_names = [m["name"] for m in results]
    num_models = len(model_names)
    aic_values = [_subjectwise_aic(m) for m in results]

    median_delta = np.full((num_models, num_models), np.nan, dtype=float)
    ci_halfwidth = np.full((num_models, num_models), np.nan, dtype=float)

    rng = np.random.default_rng(42)
    alpha = (1.0 - confidence) / 2.0

    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            diff = aic_values[i] - aic_values[j]
            valid = ~np.isnan(diff)
            if np.count_nonzero(valid) <= 1:
                continue

            diff_valid = diff[valid]
            med = float(np.median(diff_valid))
            median_delta[i, j] = med

            boot_medians = np.array([
                np.median(rng.choice(diff_valid, size=len(diff_valid), replace=True))
                for _ in range(n_bootstrap)
            ])
            lo = float(np.percentile(boot_medians, 100 * alpha))
            hi = float(np.percentile(boot_medians, 100 * (1 - alpha)))
            ci_halfwidth[i, j] = (hi - lo) / 2.0

    return (
        pd.DataFrame(median_delta, index=model_names, columns=model_names),
        pd.DataFrame(ci_halfwidth, index=model_names, columns=model_names),
    )


def bayesian_model_selection(
    results: list[dict],
    alpha0: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_samples: int = 100_000,
) -> pd.DataFrame:
    """Random-effects Bayesian model selection.

    Parameters
    ----------
    results : list[dict]
        Model summaries with ``'name'``, ``'fitness'`` (per-subject NLL),
        and ``'free'``.
    alpha0 : float
        Dirichlet prior concentration per model (1.0 = uniform).
    max_iter : int
        Maximum variational Bayes iterations.
    tol : float
        Convergence tolerance on Dirichlet parameters.
    n_samples : int
        Monte Carlo samples for exceedance probabilities.

    Returns
    -------
    pd.DataFrame
        Columns: Model, Expected Frequency, Exceedance Probability,
        Protected XP, Alpha.

    Notes
    -----
    Implements the variational Bayes algorithm of Stephan et al. (2009)
    with protected exceedance probability from Rigoux et al. (2014).
    Per-subject log model evidence is approximated as -0.5 * AIC_s.

    """
    from scipy.special import digamma, gammaln

    model_names = [m["name"] for m in results]
    K = len(results)
    N = len(results[0]["fitness"])

    log_evidence = np.zeros((N, K))
    for k, model in enumerate(results):
        log_evidence[:, k] = -0.5 * _subjectwise_aic(model)

    alpha = np.ones(K) * alpha0

    for _ in range(max_iter):
        log_u = log_evidence + digamma(alpha) - digamma(np.sum(alpha))
        log_u -= np.max(log_u, axis=1, keepdims=True)
        u = np.exp(log_u)
        r = u / np.sum(u, axis=1, keepdims=True)

        alpha_new = alpha0 + np.sum(r, axis=0)

        if np.max(np.abs(alpha_new - alpha)) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    expected_freq = alpha / np.sum(alpha)

    rng = np.random.default_rng(0)
    samples = rng.dirichlet(alpha, size=n_samples)
    winners = np.argmax(samples, axis=1)
    xp = np.array([float(np.mean(winners == k)) for k in range(K)])

    log_r = np.log(r + 1e-300)
    free_energy = float(
        np.sum(r * (log_evidence + digamma(alpha) - digamma(np.sum(alpha)) - log_r))
        + gammaln(K * alpha0)
        - K * gammaln(alpha0)
        - gammaln(np.sum(alpha))
        + np.sum(gammaln(alpha))
    )
    log_p_h0 = N * np.log(1.0 / K)
    bor = float(np.exp(log_p_h0 - np.logaddexp(log_p_h0, free_energy)))

    pxp = bor * (1.0 / K) + (1.0 - bor) * xp

    return pd.DataFrame({
        "Model": model_names,
        "Expected Frequency": expected_freq,
        "Exceedance Probability": xp,
        "Protected XP": pxp,
        "Alpha": alpha,
    })
