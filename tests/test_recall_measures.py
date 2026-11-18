"""Small counted examples of recall rates; zeros are padding, never items."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.dataset import make_dataset
from jaxcmr.analyses.srac import srac
from jaxcmr.analyses.repsrac import repsrac
from jaxcmr.analyses.intrusion_error_rate import intrusion_error_rate
from jaxcmr.analyses.omission_error_rate import omission_error_rate
from jaxcmr.analyses.order_error_rate import order_error_rate
from jaxcmr.analyses.termination_probability import conditional_termination_probability
from jaxcmr.analyses.spc import spc, fixed_pres_spc
from jaxcmr.analyses.pnr import conditional_pnr_with_repeats, pnr
from jaxcmr.analyses.nth_item_recall import (
    simple_nth_item_recall_curve,
    conditional_nth_item_recall_curve,
)
from jaxcmr.analyses.relative_srac import relative_srac
from jaxcmr.analyses.rpl import recall_probability_by_lag
from jaxcmr.analyses.corecall import conditional_corec_by_lag
from jaxcmr.analyses.corecall import corec_by_lag
from jaxcmr.analyses.corecall import conditional_corec_by_cat
from jaxcmr.analyses.cat_recall_rate import cat_recall_rate
from jaxcmr.analyses.cat_spc import cat_spc
from jaxcmr.analyses.factors import cat_factor_spc
from jaxcmr.analyses.factors import cat_recall_by_factor


@pytest.mark.parametrize(
    "measure,recalls,presentations,kwargs,expected",
    [
        pytest.param(
            srac, [[1, 1, 3, 0]], [[1, 1, 2, 2]], {}, [1, 1, 1, 0], id="serial-repeated-error"
        ),
        pytest.param(
            repsrac,
            [[1, 2, 0, 0], [1, 2, 1, 2]],
            [[1, 2, 1, 2]] * 2,
            {"size": 3},
            [1, 0.5, 0],
            id="accuracy-by-occurrence",
        ),
        pytest.param(
            intrusion_error_rate,
            [[1, 99, 2, 0], [1, 3, 2, 0]],
            [[1, 2, 3, 4]] * 2,
            {},
            [0, 0.5, 0, 0],
            id="intrusions",
        ),
        pytest.param(
            omission_error_rate,
            [[1, 2, 3, 4], [1, 0, 3, 0]],
            [[1, 2, 3, 4]] * 2,
            {},
            [0, 0.5, 0, 0.5],
            id="omissions",
        ),
        pytest.param(
            order_error_rate,
            [[2, 1, 3, 4], [1, 2, 3, 4]],
            [[1, 2, 3, 4]] * 2,
            {},
            [0.5, 0.5, 0, 0],
            id="order-errors",
        ),
        pytest.param(
            conditional_termination_probability,
            [[1, 2, 3, 4], [1, 0, 0, 0]],
            [[1, 2, 3, 4]] * 2,
            {},
            [0, 0.5, 0, 1],
            id="termination-hazard",
        ),
        pytest.param(
            spc, [[1, 2, 0], [2, 3, 0]], [[1, 2, 3]] * 2, {}, [0.5, 1.0, 0.5], id="serial-position"
        ),
        pytest.param(
            pnr, [[1, 2, 0], [2, 1, 0]], [[1, 2, 3]] * 2, {}, [0.5, 0.5, 0.0], id="first-recall"
        ),
        pytest.param(
            pnr,
            [[1, 0], [1, 0]],
            [[1, 2, 1]] * 2,
            {"size": 2},
            [1.0, 0.0, 1.0],
            id="repeated-study-positions",
        ),
        pytest.param(
            simple_nth_item_recall_curve,
            [[1, 2, 3], [2, 1, 3]],
            [[1, 2, 3]] * 2,
            {},
            [0.5, 0.5, 0.0],
            id="item-output-position",
        ),
        pytest.param(
            conditional_nth_item_recall_curve,
            [[1, 0, 0], [2, 1, 3]],
            [[1, 2, 3]] * 2,
            {},
            [0.5, 1.0, np.nan],
            id="item-opportunities",
        ),
        pytest.param(
            conditional_nth_item_recall_curve,
            [[2, 3, 4, 1], [4, 3, 2, 1]],
            [[1, 2, 3, 4]] * 2,
            {},
            [0.0, 0.0, 0.0, 1.0],
            id="last-available-item",
        ),
        pytest.param(
            conditional_nth_item_recall_curve,
            [[2, 3, 0], [1, 3, 4]],
            [[1, 2, 3]] * 2,
            {},
            [0.5, 0.0, 0.0],
            id="stop-at-current-slot",
        ),
        pytest.param(
            relative_srac,
            [[1, 2, 3], [1, 2, 1]],
            [[1, 2, 3]] * 2,
            {},
            [1.0, 1.0, 0.5],
            id="relative-serial-accuracy",
        ),
        pytest.param(
            recall_probability_by_lag,
            [[1, 0, 0]],
            [[1, 2, 1, 3]],
            {"max_lag": 2},
            [0.0, np.nan, 1.0, np.nan],
            id="repetition-lag",
        ),
        pytest.param(
            conditional_corec_by_lag,
            [[1, 2, 3, 4]],
            [[1, 2, 3, 4]],
            {},
            [1.0, 1.0, 1.0],
            id="conditional-corecall-full",
        ),
        pytest.param(
            conditional_corec_by_lag,
            [[1, 0, 0, 0]],
            [[1, 2, 3, 4]],
            {},
            [0.0, 0.0, 0.0],
            id="conditional-corecall-single",
        ),
        pytest.param(
            corec_by_lag,
            [[1, 3, 0, 0]],
            [[1, 2, 3, 4]],
            {},
            [0.0, 0.5, 0.0],
            id="joint-corecall-denominators",
        ),
        pytest.param(
            corec_by_lag,
            [[0, 0, 0, 0]],
            [[1, 2, 3, 4]],
            {},
            [0.0, 0.0, 0.0],
            id="joint-corecall-empty",
        ),
    ],
)
def test_recall_measure_known_answers(measure, recalls, presentations, kwargs, expected):
    data = make_dataset(jnp.array(recalls), pres_itemnos=jnp.array(presentations))
    np.testing.assert_allclose(measure(data, **kwargs), expected, atol=1e-6, equal_nan=True)
    if measure is spc:
        np.testing.assert_allclose(fixed_pres_spc(data["recalls"], 3), expected)


@pytest.mark.parametrize(
    "measure,recalls,kwargs,expected",
    [
        pytest.param(
            cat_spc,
            [[1, 0, 0, 0], [3, 0, 0, 0]],
            {},
            [0.5, np.nan, 0.5, np.nan],
            id="category-serial-position",
        ),
        pytest.param(
            cat_factor_spc,
            [[1, 2, 0, 0], [3, 4, 0, 0]],
            {"factor_field": "LateLPP"},
            [2.0, np.nan, 6.0, np.nan],
            id="category-factor-position",
        ),
        pytest.param(
            cat_recall_by_factor,
            [[1, 2, 0, 0], [3, 4, 0, 0]],
            {"factor_field": "LateLPP", "bin_edges": jnp.array([2.0, 5.0, 8.0])},
            [0.5, 0.5],
            id="category-factor-bin-rates",
        ),
        pytest.param(
            conditional_corec_by_cat, [[1, 2, 3, 0]] * 2, {}, [1.0, 0.5], id="category-corecall"
        ),
    ],
)
def test_category_measure_known_answers(measure, recalls, kwargs, expected):
    data = make_dataset(jnp.array(recalls))
    data.update(
        condition=jnp.array([[1, 2, 1, 2]] * 2), LateLPP=jnp.array([[2.0, 4.0, 6.0, 8.0]] * 2)
    )
    result = measure(data, category_field="condition", category_value=1, **kwargs)
    np.testing.assert_allclose(result, expected, atol=1e-6, equal_nan=True)


def test_category_rates_count_positions_once_and_exclude_padding():
    data = make_dataset(jnp.array([[1, 1, 0, 0]]), pres_itemnos=jnp.array([[1, 2, 0, 0]]))
    data["condition"] = jnp.array([[1, 1, 1, 1]])
    assert cat_recall_rate(data, "condition", 1, valid_field="pres_itemnos") == 0.5
    assert np.isnan(cat_recall_rate(data, "condition", 2, valid_field="pres_itemnos"))


def test_position_codes_recall_all_presentations_without_double_counting():
    # Item 7 appears twice; the repeated response names the second occurrence.
    data = make_dataset(jnp.array([[3, 3, 0]]), pres_itemnos=jnp.array([[7, 2, 7, 0]]), listLength=3)
    data["condition"] = jnp.ones((1, 4), dtype=int)
    np.testing.assert_allclose(spc(data), [1, 0, 1, np.nan], equal_nan=True)
    np.testing.assert_allclose(cat_spc(data, "condition", 1), [1, 0, 1, np.nan], equal_nan=True)
    assert cat_recall_rate(data, "condition", 1) == pytest.approx(2 / 3)
    np.testing.assert_allclose(simple_nth_item_recall_curve(data), [1, 1, 0])
    # Code 3 is valid even though it is not one of the item identities 7 and 2.
    np.testing.assert_allclose(intrusion_error_rate(data), [0, 0, 0, 0], equal_nan=True)
    np.testing.assert_allclose(corec_by_lag(data), [0, np.nan, np.nan], equal_nan=True)


def test_rpl_pools_item_opportunities_with_explicit_bins():
    import jax
    from jaxcmr.analyses.rpl import binned_recall_probability_by_lag, test_rpl_slope as slope_test

    data = make_dataset(
        jnp.array([[3, 3, 0], [0, 0, 0], [4, 0, 0]]),
        pres_itemnos=jnp.array([[7, 2, 7, 3, 0], [8, 9, 8, 4, 0], [9, 2, 3, 9, 0]]),
        listLength=4,
    )
    # Spacing 1: one of two items recalled. Spacing 2: one of one recalled.
    # Pooling gives 2/3, rather than averaging the two rates to obtain 3/4.
    fn = jax.jit(lambda d: binned_recall_probability_by_lag(d, ((0, 0), (1, 2))))
    np.testing.assert_allclose(fn(data), [0, np.nan, 2 / 3], equal_nan=True)
    # The full curve retains its overflow bucket: all spacings >= 1 enter the last column.
    np.testing.assert_allclose(recall_probability_by_lag(data, max_lag=1), [0, np.nan, 2 / 3], equal_nan=True)
    # First column is the singleton rate, which must not enter the spacing slope.
    slopes = slope_test(np.array([[.9, .2, .4, .6], [.1, .1, .3, .5]]), np.array([0, 2, 4]))
    assert slopes.iloc[0]["mean_left"] == pytest.approx(.1)


def test_recall_in_last_occupied_slot_after_a_gap():
    # Three items in four slots. The first response is the photo in slot four.
    data = make_dataset([4, 1, 0], pres_itemnos=[1, 0, 2, 3], listLength=3)
    np.testing.assert_allclose(spc(data), [1, np.nan, 0, 1], equal_nan=True)
    np.testing.assert_allclose(simple_nth_item_recall_curve(data, 3), [1, 0, 0])
    np.testing.assert_allclose(conditional_nth_item_recall_curve(data, 3), [1, np.nan, np.nan])
    np.testing.assert_allclose(conditional_pnr_with_repeats(data, 1, 0), [0, np.nan, 0, 1])
