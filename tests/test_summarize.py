import numpy as np
from scipy.stats import t as t_dist, ttest_rel
from jaxcmr.summarize import (
    add_summary_lines,
    calculate_ci,
    generate_t_p_matrices,
    summarize_parameters,
)


def _model_results():
    return [
        {
            "name": "ModelA",
            "fitness": [10.0, 12.0, 11.0],
            "free": ["p1", "p2"],
            "fixed": {"p3": 0.5},
            "fits": {"p1": [0.1, 0.2, 0.3], "p2": [0.4, 0.5, 0.6], "p3": [0.5, 0.5, 0.5]},
        },
        {
            "name": "ModelB",
            "fitness": [9.0, 11.0, 10.0],
            "free": ["p1"],
            "fixed": {"p2": 0.5, "p3": 0.5},
            "fits": {"p1": [0.1, 0.2, 0.3], "p2": [0.5, 0.5, 0.5], "p3": [0.5, 0.5, 0.5]},
        },
    ]


def test_calculate_ci_exact_halfwidth():
    """CI half-width matches hand-calculated t * stderr."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stderr = np.std(data, ddof=1) / np.sqrt(5)
    expected_ci = stderr * t_dist.ppf(0.975, 4)
    ci = calculate_ci(data)
    assert np.isclose(ci, expected_ci, atol=1e-10)
    assert np.isclose(ci, 1.96324, atol=0.0001)


def test_generate_t_p_matrices_exact_t_stat():
    """Off-diagonal t-stat matches scipy ttest_rel."""
    results = [
        {"name": "ModelX", "fitness": [10.0, 13.0, 11.0], "free": ["p1"]},
        {"name": "ModelY", "fitness": [9.0, 11.0, 10.0], "free": ["p1"]},
    ]
    expected_t, expected_p = ttest_rel([9.0, 11.0, 10.0], [10.0, 13.0, 11.0], alternative="less")
    t_df, p_df = generate_t_p_matrices(results)
    for name in ["ModelX", "ModelY"]:
        assert t_df.loc[name, name] == ""
        assert p_df.loc[name, name] == ""
    assert np.isclose(t_df.loc["ModelY", "ModelX"], expected_t, atol=1e-10)
    assert np.isclose(p_df.loc["ModelY", "ModelX"], expected_p, atol=1e-10)
    assert np.isclose(expected_t, -4.0, atol=1e-10)


def test_summarize_parameters_exact_mean_values():
    """Markdown table contains exact formatted mean values."""
    results = _model_results()
    table = summarize_parameters(results)
    assert "| Parameter |" in table
    assert "|---|---" in table
    assert "11.00" in table
    assert "10.00" in table
    assert "0.50" in table
    assert "12.00" in table
    assert "9.00" in table


def test_add_summary_lines_exact_formatted_values():
    """Summary rows contain exact formatted statistics."""
    md_table = ""
    errors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = add_summary_lines(md_table, errors, "test_metric")
    assert "test metric" in result
    assert "2.00" in result
    assert "5.00" in result
    assert "1.00" in result
    assert "4.00" in result
    assert "3.00" in result
    assert "6.00" in result
    lines = result.strip().split("\n")
    assert len(lines) == 3
    assert "mean" in lines[0]
    assert "min" in lines[1]
    assert "max" in lines[2]
