"""Paired contrasts must preserve subject differences and the selected axis."""

import numpy as np
import pytest
from jaxcmr.analyses import repcrp, repaccesscrp, repneighborcrp, reproutecrp, repcuecrp


@pytest.mark.parametrize(
    "compare,shape,cell,kwargs",
    [
        (repcrp.test_rep_crp_vs_control, (3, 2, 5), (0, 3), {}),
        (repaccesscrp.test_rep_access_crp_vs_control, (3, 2, 5), (0, 3), {}),
        (repneighborcrp.test_rep_neighbor_crp_vs_control, (3, 5), (3,), {"direction": "both"}),
        (reproutecrp.test_rep_route_crp_vs_control, (3, 5), (3,), {"direction": "same"}),
        (
            repcuecrp.test_rep_cue_crp_vs_control,
            (3, 2, 5, 5),
            (0, 3, 3),
            {"cue_offset": 1, "max_offset": 2},
        ),
        (repcuecrp.test_rep_cue_access_vs_control, (3, 2, 5, 5), (0, 3, 1), {"max_offset": 2}),
    ],
)
def test_paired_transition_contrasts(compare, shape, cell, kwargs):
    observed, control = np.zeros(shape), np.zeros(shape)
    observed[(slice(None),) + cell] = [0.8, 0.9, 0.7]
    control[(slice(None),) + cell] = [0.3, 0.2, 0.4]
    result = compare(observed, control, max_lag=2, **kwargs)
    if isinstance(result, dict):
        result = result["First Presentation"]
    # Differences .5,.7,.3: mean=.5, sample SD=.2, df=2.
    t = 0.5 * np.sqrt(3) / 0.2
    assert result.mean_diffs[3] == pytest.approx(0.5)
    assert result.t_stats[3] == pytest.approx(t)
    assert result.t_pvals[3] == pytest.approx(1 - t / np.sqrt(t * t + 2))


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
    assert result.mean_diffs[3] == pytest.approx(0.5)
    assert result.t_stats[3] == pytest.approx(0.5 * np.sqrt(3) / 0.2)
