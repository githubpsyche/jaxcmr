"""Exact model-comparison outputs, including the legacy likelihood conventions."""

import numpy as np
import pytest
from jaxcmr import summarize as summary


def fits():
    return [
        dict(
            name="A",
            fitness=[10.0, 12.0, 11.0],
            free=["p", "q"],
            fixed={},
            fits={"p": [0.1, 0.2, 0.3], "q": [0.4, 0.5, 0.6]},
        ),
        dict(
            name="B", fitness=[9.0, 11.0, 10.0], free=["p"], fixed={}, fits={"p": [0.1, 0.2, 0.3]}
        ),
    ]


@pytest.mark.parametrize(
    "measure,args,column,expected",
    [
        (summary.calculate_aic, (), "AIC", [-62.0, -58.0]),
        (summary.calculate_bic_scores, (), "BIC", [2 * np.log(3) - 66, np.log(3) - 60]),
        (
            summary.calculate_aic_weights,
            (),
            "AICw",
            [np.exp(-6) / (1 + np.exp(-6)), 1 / (1 + np.exp(-6))],
        ),
        (summary.calculate_aicc, ([20.0] * 3,), "AICc", [78 + 36 / 17, 66 + 12 / 18]),
        (
            summary.calculate_aicc_weights,
            ([20.0] * 3,),
            "AICcw",
            [
                1 / (1 + np.exp((12 + 36 / 17 - 12 / 18) / 2)),
                1 / (1 + np.exp(-(12 + 36 / 17 - 12 / 18) / 2)),
            ],
        ),
    ],
)
def test_information_criteria_known_values(measure, args, column, expected):
    # The legacy AIC/BIC entry points take log likelihood; AIC weights/AICc take NLL.
    table = measure(fits(), *args).set_index("Model")
    np.testing.assert_allclose(table.loc[["A", "B"], column], expected, rtol=1e-8)


@pytest.mark.parametrize(
    "measure,args",
    [
        (summary.winner_comparison_matrix, ()),
        (summary.raw_winner_comparison_matrix, ()),
        (summary.aicc_winner_comparison_matrix, ([20.0] * 3,)),
    ],
)
def test_winner_fractions(measure, args):
    result = measure(fits(), *args)
    np.testing.assert_allclose(
        result.loc[["A", "B"], ["A", "B"]], [[np.nan, 0], [1, np.nan]], equal_nan=True
    )


@pytest.mark.parametrize(
    "measure,args,difference",
    [
        (summary.pairwise_aic_differences, (), 4.0),
        (summary.pairwise_aicc_differences, ([20.0] * 3,), 4 + 12 / 17 - 4 / 18),
    ],
)
def test_paired_criterion_differences(measure, args, difference):
    mean, interval, equivalent = measure(fits(), *args)
    np.testing.assert_allclose(
        mean.loc[["A", "B"], ["A", "B"]],
        [[np.nan, difference], [-difference, np.nan]],
        equal_nan=True,
    )
    assert interval.loc["A", "B"] == pytest.approx(0.0)
    assert not equivalent.loc["A", "B"]
    if args:
        median, _ = summary.pairwise_median_aicc_differences(fits(), *args)
        assert median.loc["A", "B"] == pytest.approx(difference)
        with pytest.raises(ValueError, match="AICc undefined"):
            summary.calculate_aicc(fits(), [3.0, 20.0, 20.0])


@pytest.mark.parametrize("pairs", [[["Parent", "Child"]], ["('Parent', 'Child')"]])
def test_nested_fit_flooring(pairs):
    results = [
        dict(name="Parent", fitness=[1.0, 2.0, 3.0]),
        dict(name="Child", fitness=[4.0, 1.0, 5.0]),
    ]
    assert summary.floor_nested_fitness(results, pairs) == 2
    assert results[1]["fitness"] == [1.0, 1.0, 3.0]
