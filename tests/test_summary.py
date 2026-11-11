"""Model-comparison answers, pairing, and preservation of raw fitted results."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from jaxcmr import summarize as summary


def fits():
    return [
        dict(
            name="A",
            fitness=[10.0, 12.0, 11.0],
            free=["p", "q"],
            fixed={},
            fits={"subject": [1, 2, 3], "p": [0.1, 0.2, 0.3], "q": [0.4, 0.5, 0.6]},
        ),
        dict(
            name="B",
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


def test_pooled_fit_has_one_penalty_and_accepts_scalar_observation_count():
    # These are summed NLLs for one shared parameter vector, not three fitted rows.
    records = [
        dict(name="A", fitness=[33.0], free=["p", "q"]),
        dict(name="B", fitness=[30.0], free=["p"]),
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
            subjects=[3, 1, 2],
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
    b["fits"]["subject"] = [3, 1, 2]
    # Paired differences are 1,1,4: mean 2, SEM 1, t(.975,2)=4.30265273.
    mean, halfwidth = summary.pairwise_cv_differences([a, b])
    assert mean.loc["A", "B"] == pytest.approx(2.0)
    assert halfwidth.loc["A", "B"] == pytest.approx(4.302652729696142)


def test_observation_counts_remain_attached_to_first_records_subjects():
    records = fits()
    expected = summary.pairwise_aicc_differences(records, [10, 20, 30])
    shuffled = deepcopy(records)
    shuffled[0]["fitness"] = [11.0, 10.0, 12.0]
    shuffled[0]["fits"]["subject"] = [3, 1, 2]
    shuffled[1]["fitness"] = [11.0, 10.0, 9.0]
    shuffled[1]["fits"]["subject"] = [2, 3, 1]
    actual = summary.pairwise_aicc_differences(shuffled, [30, 10, 20])
    pd.testing.assert_frame_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "replacement,error",
    [
        ({"subject": [1, 1, 3]}, "duplicate"),
        ({"subject": [1, 2, 4]}, "same subject set"),
        ({"subject": [1, 2]}, "match the score rows"),
        ({}, "identifiers are required"),
        ({"subject": [1, 2, 3], "subjects": [3, 2, 1]}, "conflicting"),
    ],
)
def test_ambiguous_participant_pairings_fail(replacement, error):
    records = fits()
    records[1].pop("fits")
    records[1].update(replacement)
    with pytest.raises(ValueError, match=error):
        summary.winner_comparison_matrix(records)


@pytest.mark.parametrize(
    "values", [[10.0, np.nan, 11.0], [10.0, np.inf, 11.0], [[10.0], [12.0], [11.0]]]
)
def test_invalid_scores_cannot_be_counted_as_losses_or_broadcast(values):
    records = fits()
    records[0]["fitness"] = values
    with pytest.raises(ValueError, match="finite 1D"):
        summary.winner_comparison_matrix(records)


@pytest.mark.parametrize(
    "counts,error",
    [
        (20, "one positive integer"),
        ([[20], [20], [20]], "one positive integer"),
        ([20, 20.5, 20], "one positive integer"),
        ([20, np.inf, 20], "one positive integer"),
        ([20, 0, 20], "one positive integer"),
        ([20, 3, 20], "AICc undefined"),
    ],
)
def test_invalid_observation_counts_fail(counts, error):
    with pytest.raises(ValueError, match=error):
        summary.calculate_aicc(fits(), counts)


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
    records[0]["subject"].reverse()
    records[0]["fitness"].reverse()
    reordered = measure(records, *args)
    pd.testing.assert_frame_equal(reordered[1], lower)
    pd.testing.assert_frame_equal(reordered[2], upper)


@pytest.mark.parametrize("pair", [("A", "B"), ["A", "B"], "('A', 'B')"])
def test_nested_adjustments_keep_raw_fits_and_parameters_intact(pair):
    records = fits()
    records[1]["fitness"] = [12.0, 9.0, 14.0]
    records[1]["fits"]["subject"] = [3, 1, 2]
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
    table = summary.bayesian_model_selection(records, alpha0=prior, n_obs=counts)
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
    for record in records:
        record["fitness"] = (
            np.asarray(record["fitness"]) + np.arange(len(evidence)) * 100
        ).tolist()
    shifted = summary.bayesian_model_selection(records, alpha0=prior, n_obs=counts)
    np.testing.assert_allclose(shifted.iloc[:, 1:], table.iloc[:, 1:], atol=1e-8)


def test_bms_reports_nonconvergence():
    with pytest.raises(RuntimeError, match="did not converge"):
        summary.bayesian_model_selection(fits(), max_iter=1)


def test_pooled_records_are_not_random_effects_participants():
    records = [
        dict(name=name, free=[], fitness=[nll], subject=[-1])
        for name, nll in [("A", 1), ("B", 2)]
    ]
    with pytest.raises(ValueError, match="pooled fit"):
        summary.bayesian_model_selection(records)
