"""Paired contrasts must preserve subject differences and the selected axis."""

import numpy as np
import pytest
from jaxcmr.analyses import repcrp, repaccesscrp, reproutecrp, repcuecrp





@pytest.mark.parametrize(
    "compare,shape,cell,kwargs",
    [
        (repcrp.test_first_second_bias, (3, 2, 5), (3,), {}),
        (repaccesscrp.test_first_second_bias, (3, 2, 5), (3,), {}),
        (reproutecrp.test_same_switch_bias, (3, 2, 5), (3,), {}),
        (
            repcuecrp.test_first_second_bias,
            (3, 2, 5, 5),
            (3, 3),
            {"cue_offset": 1, "max_offset": 2},
        ),
        (repcuecrp.test_first_second_access_bias, (3, 2, 5, 5), (3, 1), {"max_offset": 2}),
    ],
)
def test_first_second_contrast_subtracts_control_bias(compare, shape, cell, kwargs):
    observed, control = np.zeros(shape), np.zeros(shape)
    observed[(slice(None), 0) + cell] = [0.8, 0.9, 0.7]
    observed[(slice(None), 1) + cell] = [0.3, 0.2, 0.4]
    control[(slice(None), slice(None)) + cell] = 0.4
    result = (
        compare(observed[:, 0], observed[:, 1], control[:, 0], control[:, 1], max_lag=2)
        if compare is reproutecrp.test_same_switch_bias
        else compare(observed, control, max_lag=2, **kwargs)
    )
    assert result.loc[1, "mean_difference"] == pytest.approx(0.5)
    assert result.loc[1, "t_stat"] == pytest.approx(0.5 * np.sqrt(3) / 0.2)


def test_access_contrast_selects_the_negative_offset_diagonal():
    observed, control = np.zeros((3, 2, 5, 5)), np.zeros((3, 2, 5, 5))
    observed[:, 0, 3, 1] = [.8, .9, .7]
    control[:, 0, 3, 1] = [.3, .2, .4]
    result = repcuecrp.test_rep_cue_access_vs_control(observed, control, max_offset=2, max_lag=2)
    assert result.loc[(1, 1), "mean_difference"] == pytest.approx(.5)
    assert result.loc[(1, -1), "mean_difference"] == 0
