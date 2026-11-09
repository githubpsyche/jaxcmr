"""Repetition access, serial run-up, and routes have different denominators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.analyses import (
    repaccesscrp,
    serialrepcrp,
    reproutecrp,
    repneighborcrp,
    compound_cueing_crp,
)
from jaxcmr.helpers import make_dataset


@pytest.mark.parametrize(
    "present,recalls,min_lag,cells",
    [
        ([1, 2, 3, 4, 5, 1, 6, 7], [0, 2, 1, 0], 4, [(0, 1), (1, -4)]),
        ([1, 2, 3, 4, 5, 1, 6, 7], [7, 1, 0, 0], 4, [(0, 6), (1, 1)]),
        ([1, 2, 3, 4, 5, 1, 6, 7], [1, 2, 0, 0], 4, []),
        ([1, 2, 3, 4, 5, 1, 6, 7], [0, 0, 0, 0], 4, []),
        ([1, 2, 1, 2], [1, 2, 0, 0], 0, [(0, -1), (0, 1), (1, -3), (1, -1)]),
    ],
)
def test_repeater_access_counts_available_occurrences(present, recalls, min_lag, cells):
    actual, possible = repaccesscrp.tabulate_trial(
        jnp.array(recalls), jnp.array(present), min_lag=min_lag, size=2
    )
    expected = np.zeros((2, 2 * len(present) - 1), dtype=int)
    for occurrence, lag in cells:
        expected[occurrence, len(present) - 1 + lag] = 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(possible, expected)


@pytest.mark.parametrize(
    "recalls,cells",
    [([1, 2, 0, 0], [(0, 1), (1, -2)]), ([1, 5, 2, 0], [(0, 4), (1, 1)]), ([2, 1, 4, 0], [])],
)
def test_serial_runup_scores_error_transition_then_blocks(recalls, cells):
    actual, possible = serialrepcrp.tabulate_trial(
        jnp.array(recalls), jnp.array([1, 2, 3, 1, 4, 5, 6, 7]), min_lag=2, size=2
    )
    expected = np.zeros((2, 15), dtype=int)
    for occurrence, lag in cells:
        expected[occurrence, 7 + lag] = 1
    np.testing.assert_array_equal(actual, expected)
    if cells:
        np.testing.assert_array_equal(possible[:, [4, 7, 10]], [[0, 1, 1], [1, 1, 1]])
    else:
        assert not np.any(possible)


@pytest.mark.parametrize(
    "recalls,lag2,entry,target_lags",
    [
        ([2, 3, 4], False, 0, [1, -4]),
        ([2, 3, 9], False, 0, [6, 1]),
        ([7, 3, 9], False, 1, [6, 1]),
        ([7, 3, 4], False, 1, [1, -4]),
        ([1, 3, 4], True, 0, [1, -4]),
        ([6, 3, 9], True, 1, [6, 1]),
    ],
)
def test_routes_distinguish_entry_and_exit_occurrences(recalls, lag2, entry, target_lags):
    present = jnp.array([2, 3, 1, 4, 5, 6, 7, 1, 8])
    actual, possible = reproutecrp.tabulate_trial(
        jnp.array(recalls + [0]), present, direction="both", use_lag2=lag2, min_lag=4
    )
    expected = np.zeros((2, 2, 17), dtype=int)
    expected[entry, 0, 8 + target_lags[0]] = 1
    expected[entry, 1, 8 + target_lags[1]] = 1
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.asarray(possible)[expected.astype(bool)] == 1)
    assert not np.any(possible[1 - entry])
    same, same_possible = reproutecrp.tabulate_trial(
        jnp.array(recalls), present, direction="same", use_lag2=lag2, min_lag=4
    )
    switch, switch_possible = reproutecrp.tabulate_trial(
        jnp.array(recalls), present, direction="switch", use_lag2=lag2, min_lag=4
    )
    np.testing.assert_array_equal(same, expected[0, 0] + expected[1, 1])
    np.testing.assert_array_equal(switch, expected[0, 1] + expected[1, 0])
    np.testing.assert_array_equal(same_possible + switch_possible, possible.sum((0, 1)))
    if lag2:
        excluded, _ = reproutecrp.tabulate_trial(
            jnp.array(recalls), present, direction="both", use_lag2=False, min_lag=4
        )
        assert not np.any(excluded)


@pytest.mark.parametrize(
    "present,recalls,direction,lag2,lag",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 12], [4, 0, 0], "i2j", True, None),
        ([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 12], [0, 0, 0], "both", True, None),
        ([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 12], [12, 4], "j2i", True, None),
        ([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 12], [4, 12, 0], "i2j", True, 3),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 12], [12, 4], "j2i", False, 1),
    ],
)
def test_neighbor_padding_and_real_last_position(present, recalls, direction, lag2, lag):
    run = jax.jit(
        repneighborcrp.tabulate_trial, static_argnames=("direction", "use_lag2", "min_lag")
    )
    actual, possible = run(
        jnp.array(recalls), jnp.array(present), direction=direction, use_lag2=lag2, min_lag=4
    )
    expected = np.zeros(23, dtype=int)
    if lag is not None:
        expected[11 + lag] = 1
        assert possible[11 + lag] == 1
    else:
        assert not np.any(possible)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("direction,lag2", [("i2j", False), ("j2i", True), ("both", True)])
def test_padding_preserves_neighbor_opportunities_including_repeated_last_item(direction, lag2):
    present = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 4])
    kwargs = dict(direction=direction, use_lag2=lag2, min_lag=4)
    compact = repneighborcrp.tabulate_trial(jnp.array([10, 4, 3]), present, **kwargs)
    padded = repneighborcrp.tabulate_trial(jnp.array([0, 10, 0, 4, 0, 3, 0]), present, **kwargs)
    np.testing.assert_array_equal(padded, compact)


@pytest.mark.parametrize(
    "recalls,expected,spacing,repeated",
    [
        ([[1, 2, 3, 0], [8, 9, 3, 0], [1, 2, 4, 0], [8, 9, 4, 0]], [0.5, np.nan], 6, True),
        ([[8, 2, 3, 0], [1, 9, 3, 0], [8, 2, 4, 0], [1, 9, 4, 0]], [np.nan, 0.5], 6, True),
        ([[3, 1, 2, 4]], [np.nan, np.nan], 6, True),
        ([[2, 1, 3, 0]], [np.nan, np.nan], 6, True),
        ([[7, 3, 0, 0]], [np.nan, np.nan], 6, True),
        ([[1, 2, 3, 0]], [np.nan, np.nan], 12, True),
        ([[1, 2, 3, 0]], [np.nan, np.nan], 6, False),
    ],
)
def test_compound_cues_require_ordered_available_neighborhoods(
    recalls, expected, spacing, repeated
):
    presentation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10, 11] if repeated else list(range(1, 13))
    data = make_dataset(jnp.array(recalls), jnp.array([presentation] * len(recalls)))
    np.testing.assert_allclose(
        compound_cueing_crp.compound_cueing_crp(data, min_spacing=spacing, size=2),
        expected,
        equal_nan=True,
    )
