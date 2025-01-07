from typing import Optional

import numpy as np
from scipy.stats import t


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
