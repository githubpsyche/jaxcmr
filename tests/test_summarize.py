"""Statistical summaries, model comparisons and formatted results."""
from copy import deepcopy
import pandas as pd
import pytest
from jaxcmr import summarize as summary
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
    assert calculate_ci([1., 2., np.nan, 3., 4., 5.]) == pytest.approx(ci)
    assert np.isnan(calculate_ci([2., np.nan]))
    assert np.isnan(calculate_ci([]))


def test_generate_t_p_matrices_exact_t_stat():
    """Off-diagonal t-stat matches scipy ttest_rel."""
    results = [
        {"subject": [1, 2, 3], "name": "ModelX", "fitness": [10.0, 13.0, 11.0], "free": ["p1"]},
        {"subject": [1, 2, 3], "name": "ModelY", "fitness": [9.0, 11.0, 10.0], "free": ["p1"]},
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
    table = summarize_parameters(_model_results())
    rows = [tuple(cell.strip() for cell in line.strip('|').split('|'))
            for line in table.strip().splitlines()]
    assert rows[:2] == [('Parameter', 'Statistic', 'ModelA', 'ModelB'), ('---',) * 4]
    assert [row[0] for row in rows[2:] if row[0]] == ["fitness", "p1", "p2", "p3"]
    parameter = None
    values = {}
    for label, statistic, *model_values in rows[2:]:
        parameter = label or parameter
        for model, value in zip(rows[0][2:], model_values, strict=True):
            key = (parameter, statistic, model)
            assert key not in values
            values[key] = value
    # Means/minima/maxima of the three explicit rows in _model_results.
    expected = {
        'fitness': [('11.00', '10.00'), ('10.00', '9.00'), ('12.00', '11.00')],
        'p1': [('0.20', '0.20'), ('0.10', '0.10'), ('0.30', '0.30')],
        'p3': [('0.50', '0.50')] * 3,
        'p2': [('0.50', '0.50'), ('0.40', '0.50'), ('0.60', '0.50')],
    }
    assert values == {
        (parameter, statistic, model): value
        for parameter, pairs in expected.items()
        for statistic, pair in zip(('mean', 'min', 'max'), pairs, strict=True)
        for model, value in zip(('ModelA', 'ModelB'), pair, strict=True)
    }, 'Summary values must remain attached to their model and statistic'


def test_add_summary_lines_exact_formatted_values():
    result = add_summary_lines('', [[1., 2., 3.], [4., 5., 6.]], 'test_metric')
    rows = [tuple(cell.strip() for cell in line.strip('|').split('|'))
            for line in result.strip().splitlines()]
    assert rows == [
        ('test metric', 'mean', '2.00', '5.00'),
        ('', 'min', '1.00', '4.00'),
        ('', 'max', '3.00', '6.00'),
    ]


def test_parameter_summary_uses_nonmissing_values_and_marks_unavailable_ci():
    record = {
        "name": "A", "fitness": [1., 2., 3.],
        "fits": {"subject": [1, 2, 3], "p": [1., np.nan, 3.],
                 "single": [np.nan, 2., np.nan], "absent": [np.nan] * 3},
    }
    before = deepcopy(record)
    table = summarize_parameters([record], include_std=True, include_ci=True)
    # For p: mean=2, population SD=1, SEM=1, t(.975,1)=12.7062.
    assert "| p | mean | 2.00 +/- 12.71 |" in table
    assert "| p | mean | 2.00 +/- 12.71 |\n|  | std | 1.00 |\n|  | min | 1.00 |\n|  | max | 3.00 |" in table
    assert "| single | mean | 2.00 (CI unavailable) |" in table
    assert "| absent | mean |  |\n|  | std |  |\n|  | min |  |\n|  | max |  |" in table
    assert "| subject |" not in table
    requested = summarize_parameters([record], query_parameters=["single", "missing", "p"])
    assert [line.split("|")[1].strip() for line in requested.splitlines()[2::3]] == [
        "fitness", "single", "missing", "p"
    ]
    for key in record["fits"]:
        np.testing.assert_equal(record["fits"][key], before["fits"][key])
    pooled = {"name": "Pooled", "fitness": [10.], "fits": {"subject": [-1], "p": [2.]}}
    assert "| p | mean | 2.00 (CI unavailable) |" in summarize_parameters([pooled], include_ci=True)


def fits():
    return [
        dict(
            name="A",
            subject=[1, 2, 3],
            fitness=[10.0, 12.0, 11.0],
            free=["p", "q"],
            fixed={},
            fits={"subject": [1, 2, 3], "p": [0.1, 0.2, 0.3], "q": [0.4, 0.5, 0.6]},
        ),
        dict(
            name="B",
            subject=[1, 2, 3],
            fitness=[9.0, 11.0, 10.0],
            free=["p"],
            fixed={},
            fits={"subject": [1, 2, 3], "p": [0.1, 0.2, 0.3]},
        ),
    ]


@pytest.mark.parametrize(
    "measure,args,column,expected",
    [
        (summary.calculate_aic, (), "AIC", [78.0, 66.0]),
        (
            summary.calculate_bic_scores,
            ([10, 20, 30],),
            "BIC",
            [66 + 2 * np.log(6000), 60 + np.log(6000)],
        ),
        (
            summary.calculate_aic_weights,
            (),
            "AICw",
            [1 / (1 + np.exp(6)), 1 / (1 + np.exp(-6))],
        ),
        (summary.calculate_aicc, ([20] * 3,), "AICc", [78 + 36 / 17, 66 + 12 / 18]),
        (
            summary.calculate_aicc_weights,
            ([20] * 3,),
            "AICcw",
            [
                1 / (1 + np.exp((12 + 36 / 17 - 12 / 18) / 2)),
                1 / (1 + np.exp(-(12 + 36 / 17 - 12 / 18) / 2)),
            ],
        ),
    ],
)
def test_information_criteria_known_values(measure, args, column, expected):
    # Three independently fitted parameter sets: A has six parameters, B three.
    table = measure(fits(), *args)
    assert table.iloc[0]["Model"] == "B"
    np.testing.assert_allclose(
        table.set_index("Model").loc[["A", "B"], column], expected
    )


def test_numerical_criteria_broadcast_without_fit_records():
    nll = np.array([[10., 12., 11.], [9., 11., 10.]])
    counts = np.array([[2], [1]])
    observations = np.array([10, 20, 30])
    originals = [values.copy() for values in (nll, counts, observations)]
    # Each column is a participant fit. Penalize each row by 2*k, not by a
    # combined parameter count; AICc denominators are n-k-1 for that fit.
    expected_aic = np.array([[24., 28., 26.], [20., 24., 22.]])
    np.testing.assert_array_equal(summary.aic_scores(nll, counts), expected_aic)
    np.testing.assert_allclose(
        summary.aicc_scores(nll, counts, observations),
        expected_aic + [[12/7, 12/17, 12/27], [4/8, 4/18, 4/28]],
    )
    np.testing.assert_allclose(
        summary.bic_scores(nll, counts, observations),
        [[20 + 2*np.log(10), 24 + 2*np.log(20), 22 + 2*np.log(30)],
         [18 + np.log(10), 22 + np.log(20), 20 + np.log(30)]],
    )
    for values, original in zip((nll, counts, observations), originals):
        np.testing.assert_array_equal(values, original)
    # A custom report can calculate a single pooled fit without constructing a record.
    scalar = summary.aic_scores(33., 2)
    assert scalar.shape == ()
    assert float(scalar) == 70.
    assert float(summary.aicc_scores(33., 2, 60)) == pytest.approx(70 + 12/57)
    assert float(summary.bic_scores(33., 2, 60)) == pytest.approx(66 + 2*np.log(60))


def test_pooled_fit_has_one_penalty_and_accepts_scalar_observation_count():
    # These are summed NLLs for one shared parameter vector, not three fitted rows.
    records = [
        dict(name="A", subject=[-1], fitness=[33.0], free=["p", "q"]),
        dict(name="B", subject=[-1], fitness=[30.0], free=["p"]),
    ]
    np.testing.assert_allclose(
        summary.calculate_aic(records).set_index("Model").loc[["A", "B"], "AIC"],
        [70, 62],
    )
    weights = summary.calculate_aic_weights(records).set_index("Model")
    assert weights.loc["A", "AICw"] == pytest.approx(1 / (1 + np.exp(4)))
    bic = summary.calculate_bic_scores(records, 60).set_index("Model")
    assert bic.loc["A", "BIC"] == pytest.approx(66 + 2 * np.log(60))
    aicc = summary.calculate_aicc(records, 60).set_index("Model")
    assert aicc.loc["A", "AICc"] == pytest.approx(70 + 12 / 57)


@pytest.mark.parametrize(
    "measure,args",
    [
        (summary.winner_comparison_matrix, ()),
        (summary.raw_winner_comparison_matrix, ()),
        (summary.aicc_winner_comparison_matrix, ([20] * 3,)),
        (summary.cv_winner_comparison_matrix, ()),
    ],
)
def test_winners_align_subjects_and_leave_ties_as_nonwins(measure, args):
    # At subjects 1,2,3: A has 0,1,2 and B has 0,2,1. One tie, one win each.
    records = [
        dict(
            name="A",
            fitness=[0.0, 1.0, 2.0],
            cv_fitness=[0.0, 1.0, 2.0],
            free=[],
            subject=[1, 2, 3],
        ),
        dict(
            name="B",
            fitness=[1.0, 0.0, 2.0],
            cv_fitness=[1.0, 0.0, 2.0],
            free=[],
            subject=[3, 1, 2],
        ),
    ]
    table = measure(records, *args)
    np.testing.assert_allclose(
        table, [[np.nan, 1 / 3], [1 / 3, np.nan]], equal_nan=True
    )


@pytest.mark.parametrize(
    "measure,args,expected",
    [
        (summary.pairwise_aic_differences, (), 4.0),
        (summary.pairwise_aicc_differences, ([20] * 3,), 4 + 12 / 17 - 4 / 18),
    ],
)
def test_mean_differences_return_estimate_and_interval_only(measure, args, expected):
    mean, halfwidth = measure(fits(), *args)
    np.testing.assert_allclose(
        mean, [[np.nan, expected], [-expected, np.nan]], equal_nan=True
    )
    assert halfwidth.loc["A", "B"] == pytest.approx(0.0)


def test_cv_differences_pair_by_ids_and_do_not_charge_parameters():
    a, b = fits()
    a["cv_fitness"] = [3.0, 4.0, 6.0]
    b["cv_fitness"] = [2.0, 2.0, 3.0]
    b["subject"] = [3, 1, 2]
    # Paired differences are 1,1,4: mean 2, SEM 1, t(.975,2)=4.30265273.
    mean, halfwidth = summary.pairwise_cv_differences([a, b])
    assert mean.loc["A", "B"] == pytest.approx(2.0)
    assert halfwidth.loc["A", "B"] == pytest.approx(4.302652729696142)


def test_observation_counts_remain_attached_to_first_records_subjects():
    records = fits()
    expected = summary.pairwise_aicc_differences(records, [10, 20, 30])
    shuffled = deepcopy(records)
    shuffled[0]["fitness"] = [11.0, 10.0, 12.0]
    shuffled[0]["subject"] = np.array([3, 1, 2])
    shuffled[1]["fitness"] = [11.0, 10.0, 9.0]
    shuffled[1]["subject"] = np.array([2, 3, 1])
    actual = summary.pairwise_aicc_differences(shuffled, [30, 10, 20])
    pd.testing.assert_frame_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])


def test_aicc_reports_undefined_small_sample_correction():
    with pytest.raises(ValueError, match="AICc undefined"):
        summary.calculate_aicc(fits(), [20, 3, 20])
    with pytest.raises(ValueError, match="AICc undefined"):
        summary.aicc_scores([10., 12., 11.], 2, [20, 3, 20])


@pytest.mark.parametrize(
    "measure,args",
    [
        (summary.pairwise_median_aic_differences, ()),
        (summary.pairwise_median_aicc_differences, ([20] * 5,)),
    ],
)
def test_skewed_percentile_interval_is_not_centered_on_median(measure, args):
    # Differences 0,0,2,20,100: exact bootstrap median quantiles are 0 and 100.
    records = [
        dict(
            name="A",
            subject=list(range(5)),
            free=[],
            fitness=[2.0, 2.0, 3.0, 12.0, 52.0],
        ),
        dict(name="B", subject=list(range(5)), free=[], fitness=[2.0] * 5),
    ]
    median, lower, upper = measure(records, *args)
    assert (median.loc["A", "B"], lower.loc["A", "B"], upper.loc["A", "B"]) == (
        2.0,
        0.0,
        100.0,
    )
    np.testing.assert_allclose(median, -median.T, equal_nan=True)
    np.testing.assert_allclose(lower, -upper.T, equal_nan=True)
    records[0]["subject"].reverse()
    records[0]["fitness"].reverse()
    reordered = measure(records, *args)
    pd.testing.assert_frame_equal(reordered[1], lower)
    pd.testing.assert_frame_equal(reordered[2], upper)


@pytest.mark.parametrize("pair", [("A", "B"), ["A", "B"]])
def test_nested_adjustments_keep_raw_fits_and_parameters_intact(pair):
    records = fits()
    records[1]["fitness"] = [12.0, 9.0, 14.0]
    records[1]["subject"] = [3, 1, 2]
    original = deepcopy(records)
    adjusted, changes = summary.floor_nested_fitness(records, [pair])
    assert records == original
    assert adjusted[1] == dict(
        name="B",
        free=["p"],
        fitness=[9.0, 12.0, 11.0],
        subject=[1, 2, 3],
        comparison_only=True,
    )
    assert changes.to_dict("records") == [
        dict(subject=2, parent="A", child="B", original_nll=14.0, adjusted_nll=12.0),
        dict(subject=3, parent="A", child="B", original_nll=12.0, adjusted_nll=11.0),
    ]
    assert summary.calculate_aic(adjusted).set_index("Model").loc["B", "AIC"] == 70
    adjusted[0]["free"].append("new")
    adjusted[0]["fitness"][0] = 0
    assert records == original


def test_nested_chain_uses_supplied_order_and_records_raw_child_scores():
    records = [
        dict(name=name, free=[], fitness=[score], subject=[-1])
        for name, score in [("A", 1.0), ("B", 3.0), ("C", 5.0)]
    ]
    adjusted, changes = summary.floor_nested_fitness(
        records, [("B", "C"), ("A", "B"), ("B", "C")]
    )
    assert adjusted[2]["fitness"] == [1.0]
    assert changes["original_nll"].tolist() == [5.0, 3.0, 5.0]
    assert changes["adjusted_nll"].tolist() == [3.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "evidence,prior,counts,alpha,bor",
    [
        (
            np.tile([0.0, -0.5], (10, 1)),
            1.0,
            None,
            [10.280188840951185, 1.7198111590488154],
            0.43508476444931393,
        ),
        (
            np.array(
                [
                    [0.0, -0.5, -1.0],
                    [0.0, -2.0, -0.3],
                    [-0.5, 0.0, -0.7],
                    [-1.0, -0.5, 0.0],
                    [0.0, -0.2, -0.4],
                ]
            ),
            0.5,
            [20] * 5,
            [4.247950296150237, 0.9989897914387924, 1.2530599124109707],
            0.8202238711114928,
        ),
    ],
)
def test_bms_reference_posterior_and_protection_are_offset_invariant(
    evidence, prior, counts, alpha, bor
):
    # Gold values: independent fixed-point root solve and Dirichlet KL expression,
    # checked against SPM spm_BMS_bor.m (Rigoux A.17/A.20). No reference solver shipped.
    # Construct NLL so the AIC/AICc approximation is this evidence minus 10.
    records = [
        dict(
            name=str(i),
            subject=list(range(len(evidence))),
            free=[f"p{j}" for j in range(i)],
            fitness=(
                10 - evidence[:, i] - i - (i * (i + 1) / (20 - i - 1) if counts else 0)
            ).tolist(),
        )
        for i in range(evidence.shape[1])
    ]
    original_evidence = evidence.copy()
    table = summary.random_effects_model_selection(
        evidence, [record["name"] for record in records], alpha0=prior
    )
    np.testing.assert_array_equal(evidence, original_evidence)
    approximated = summary.bayesian_model_selection(records, alpha0=prior, n_obs=counts)
    pd.testing.assert_frame_equal(approximated, table, atol=1e-8)
    np.testing.assert_allclose(table["Alpha"], alpha, atol=5e-8)
    np.testing.assert_allclose(
        table["Expected Frequency"], np.array(alpha) / sum(alpha), atol=1e-8
    )
    xp = table["Exceedance Probability"].to_numpy()
    np.testing.assert_allclose(
        table["Protected XP"], bor / len(alpha) + (1 - bor) * xp, atol=1e-8
    )
    if len(alpha) == 2:
        # Independent exact beta-tail exceedance; sampling error at 100,000 draws.
        assert xp[0] == pytest.approx(0.99677633508878, abs=0.001)
    direct_shifted = summary.random_effects_model_selection(
        evidence + np.arange(len(evidence))[:, None] * 100,
        [record["name"] for record in records], alpha0=prior,
    )
    np.testing.assert_allclose(direct_shifted.iloc[:, 1:], table.iloc[:, 1:], atol=1e-8)
    for record in records:
        record["fitness"] = (
            np.asarray(record["fitness"]) + np.arange(len(evidence)) * 100
        ).tolist()
    shifted = summary.bayesian_model_selection(records, alpha0=prior, n_obs=counts)
    np.testing.assert_allclose(shifted.iloc[:, 1:], table.iloc[:, 1:], atol=1e-8)


def test_bms_reports_nonconvergence():
    with pytest.raises(RuntimeError, match="did not converge"):
        summary.bayesian_model_selection(fits(), max_iter=1)
