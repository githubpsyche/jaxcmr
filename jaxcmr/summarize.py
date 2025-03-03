import json
from typing import Mapping, Optional

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
    "summarize_parameters",
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


def add_summary_lines(
    md_table: str,
    errors: list[list[float]],
    label: str,
    include_std=False,
    include_ci=False,
) -> str:
    """Returns markdown table added lines for mean and errors of the values.

    Args:
        md_table: markdown table to add summary lines to.
        errors: values to summarize.
        label: label for the summary lines.
    """
    # Add a line for the mean
    md_table += f"| {label.replace('_', ' ')} "
    if include_std:
        md_table += "| mean "
    for variant_values in errors:
        if np.isnan(np.mean(variant_values)):
            md_table += "| "
        elif include_ci:
            md_table += f"| {np.mean(variant_values):.2f} +/- {calculate_ci(variant_values):.2f} "
        else:
            md_table += f"| {np.mean(variant_values):.2f} "
    md_table += "|\n"

    # Add a line for the standard deviation
    if include_std:
        md_table += "| | std "
        for variant_values in errors:
            if np.isnan(np.std(variant_values)):
                md_table += "| "
                continue
            md_table += f"| {np.std(variant_values):.2f} "
        md_table += "|\n"

    # Add a line for the confidence interval, but just if label is 'fitness'
    # if label != "fitness":
    #     return md_table
    # md_table += "| | ci "
    # for variant_values in errors:
    #     md_table += f"| +/- {calculate_ci(variant_values):.2f} "
    # md_table += "|\n"
    return md_table


def summarize_parameters(
    model_data: list[dict],
    query_parameters: Optional[list[str]] = None,
    include_std=False,
    include_ci=False,
):
    """Returns markdown table summarizing the parameters across model variants.

    Computes the mean and confidence interval for each parameter across all subjects for each
    model variant, with an option to specify which parameters to include in the table and t
    their order.

    Args:
    model_data : list[dict[str, dict[str, list]]]
        A list of dictionaries with with dictionaries list values.
        Each list corresponds to a model or fitting variant.
        inner list is p
    query_parameters : list[str], optional
    """
    # Extract model names from the first entry of each model variant list
    model_names = [model_data[i]["name"] for i in range(len(model_data))]

    # identify query parameters; by default, is all unique fixed params across model variants
    if query_parameters is None:
        query_parameters = list(
            set().union(*[entry["fixed"].keys() for entry in model_data])
        )

    if include_std:
        md_table = (
            "| | | " + " | ".join([n.replace("_", " ") for n in model_names]) + " |\n"
        )
    else:
        md_table = (
            "| | " + " | ".join([n.replace("_", " ") for n in model_names]) + " |\n"
        )
    md_table += "|---" + ("|---" * (len(model_data) + 1)) + "|\n"

    # likelihood entry first
    values = [variant_data["fitness"] for variant_data in model_data]
    md_table = add_summary_lines(
        md_table, values, "fitness", include_std=include_std, include_ci=include_ci
    )

    # Compute the mean and confidence interval for params for each model variant
    for param in query_parameters:
        values = []
        for variant_data in model_data:
            try:
                values.append(variant_data["fits"][param])
            except KeyError:
                values.append([np.nan for _ in range(len(variant_data["fitness"]))])
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
        log_likelihood = sum(model["fitness"])  # assuming fitness is log-likelihood
        aic = 2 * k + 2 * log_likelihood
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


def winner_comparison_matrix(results: list[dict]) -> pd.DataFrame:
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
