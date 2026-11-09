"""Small counted examples of recall rates; zeros are padding, never items."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.helpers import make_dataset
from jaxcmr.analyses.srac import srac
from jaxcmr.analyses.repsrac import repsrac
from jaxcmr.analyses.intrusion_error_rate import intrusion_error_rate
from jaxcmr.analyses.omission_error_rate import omission_error_rate
from jaxcmr.analyses.order_error_rate import order_error_rate
from jaxcmr.analyses.termination_probability import conditional_termination_probability
from jaxcmr.analyses.spc import spc, fixed_pres_spc
from jaxcmr.analyses.pnr import pnr
from jaxcmr.analyses.nth_item_recall import (
    simple_nth_item_recall_curve,
    conditional_nth_item_recall_curve,
)
from jaxcmr.analyses.relative_srac import relative_srac
from jaxcmr.analyses.rpl import recall_probability_by_lag
from jaxcmr.analyses.conditional_corec_by_lag import conditional_corec_by_lag
from jaxcmr.analyses.joint_corec_by_lag import corec_by_lag
from jaxcmr.analyses.conditional_corec_by_cat import conditional_corec_by_cat
from jaxcmr.analyses.cat_recall_rate import cat_recall_rate
from jaxcmr.analyses.cat_spc import cat_spc
from jaxcmr.analyses.cat_lpp_spc import cat_lpp_spc
from jaxcmr.analyses.cat_lpp_by_recall import expand_categories_by_recall
from jaxcmr.analyses.cat_recall_by_lpp import cat_recall_by_lpp


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
            cat_lpp_spc,
            [[1, 2, 0, 0], [3, 4, 0, 0]],
            {"lpp_field": "LateLPP"},
            [2.0, np.nan, 6.0, np.nan],
            id="category-factor-position",
        ),
        pytest.param(
            cat_recall_by_lpp,
            [[1, 2, 0, 0], [3, 4, 0, 0]],
            {"lpp_field": "LateLPP", "bin_edges": jnp.array([2.0, 5.0, 8.0])},
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
    remap = make_dataset(jnp.array([[1, 3, 0, 0]]))
    remap["condition"] = jnp.array([[1, 1, 2, 2]])
    np.testing.assert_array_equal(expand_categories_by_recall(remap, "condition"), [[2, 1, 4, 3]])
