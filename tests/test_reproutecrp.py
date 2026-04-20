import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from jax import jit
from matplotlib.axes import Axes
from scipy import stats

from jaxcmr.analyses.reproutecrp import (
    RepRouteCRPTestResult,
    plot_rep_route_crp,
    reproutecrp,
    subject_rep_route_crp,
    tabulate_trial,
    test_rep_route_crp_vs_control as compare_rep_route_crp,
    test_same_switch_bias as compare_same_switch_bias,
)
from jaxcmr.helpers import make_dataset


def route_dataset():
    """Small dataset with i-route and j-route triples."""
    pres = jnp.array([[1, 2, 3, 4, 5, 1, 6, 7]] * 4, dtype=jnp.int32)
    recalls = jnp.array(
        [
            [2, 1, 3, 0, 0, 0, 0, 0],
            [2, 1, 6, 0, 0, 0, 0, 0],
            [6, 1, 7, 0, 0, 0, 0, 0],
            [6, 1, 3, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    subjects = jnp.array([0, 0, 1, 1])
    return make_dataset(recalls, pres, subject=subjects)


def test_tabulate_trial_exact_route_counts():
    """Behavior: tabulation counts neighbor -> repeater -> next triples.

    Given:
      - Item 1 appears at positions i=1 and j=6.
      - Recall order [2, 1, 3] is i+1 -> R -> i+2.
    When:
      - ``tabulate_trial`` is called with direction="both".
    Then:
      - Incoming route i is active.
      - The following recall is counted at lag +2 from i and lag -3 from j.
    Why this matters:
      - Verifies that the analysis conditions on the route into the repeater
        and scores the recall after the repeater.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 1, 3, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(
        trial, pres, direction="both", use_lag2=False, min_lag=4
    )

    assert actual.shape == (2, 2, 15)
    assert possible.shape == (2, 2, 15)
    assert int(actual[0, 0, lag_range + 2]) == 1
    assert int(actual[0, 1, lag_range - 3]) == 1
    assert int(actual[1].sum()) == 0
    assert int(actual.sum()) == 2
    assert int(possible[0, 0, lag_range + 2]) == 1
    assert int(possible[0, 1, lag_range - 3]) == 1
    assert int(possible[1].sum()) == 0


def test_same_and_switch_select_count_combinations():
    """Behavior: same/switch combine underlying route-count arrays.

    Given:
      - A route-conditioned triple i+1 -> R -> i+2.
    When:
      - ``tabulate_trial`` selects same or switch directions.
    Then:
      - same equals i2i + j2j counts.
      - switch equals i2j + j2i counts.
    Why this matters:
      - Verifies that same/switch are count-combined CRPs, not post-hoc
        arithmetic on already-divided CRP curves.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 1, 3, 0, 0, 0, 0, 0], dtype=jnp.int32)

    both_actual, both_possible = tabulate_trial(
        trial, pres, direction="both", use_lag2=False, min_lag=4
    )
    same_actual, same_possible = tabulate_trial(
        trial, pres, direction="same", use_lag2=False, min_lag=4
    )
    switch_actual, switch_possible = tabulate_trial(
        trial, pres, direction="switch", use_lag2=False, min_lag=4
    )

    np.testing.assert_array_equal(
        np.array(same_actual), np.array(both_actual[0, 0] + both_actual[1, 1])
    )
    np.testing.assert_array_equal(
        np.array(same_possible), np.array(both_possible[0, 0] + both_possible[1, 1])
    )
    np.testing.assert_array_equal(
        np.array(switch_actual), np.array(both_actual[0, 1] + both_actual[1, 0])
    )
    np.testing.assert_array_equal(
        np.array(switch_possible), np.array(both_possible[0, 1] + both_possible[1, 0])
    )


def test_reproutecrp_shapes_and_probability_range():
    """Behavior: route CRP returns expected shapes and valid probabilities."""
    dataset = route_dataset()

    both = reproutecrp(dataset, direction="both", use_lag2=False, min_lag=4)
    same = reproutecrp(dataset, direction="same", use_lag2=False, min_lag=4)
    switch = reproutecrp(dataset, direction="switch", use_lag2=False, min_lag=4)
    i2i = reproutecrp(dataset, direction="i2i", use_lag2=False, min_lag=4)

    assert both.shape == (2, 2, 15)
    assert same.shape == (15,)
    assert switch.shape == (15,)
    assert i2i.shape == (15,)

    finite = both[~jnp.isnan(both)]
    assert jnp.all(finite >= 0.0).item()
    assert jnp.all(finite <= 1.0).item()


def test_subject_rep_route_crp_shape():
    """Behavior: subject-level route CRP preserves subject and route axes."""
    dataset = route_dataset()
    mask = jnp.ones(4, dtype=bool)

    both = subject_rep_route_crp(
        dataset, mask, direction="both", use_lag2=False, min_lag=4, max_lag=3
    )
    same = subject_rep_route_crp(
        dataset, mask, direction="same", use_lag2=False, min_lag=4, max_lag=3
    )

    assert both.shape == (2, 2, 2, 7)
    assert same.shape == (2, 7)


def test_rep_route_crp_vs_control_exact_t_stat():
    """Behavior: t-stat matches scipy ttest_rel for controlled inputs."""
    observed = np.array(
        [
            [0, 0, 0, 0.8, 0],
            [0, 0, 0, 0.9, 0],
            [0, 0, 0, 0.7, 0],
        ]
    )
    control = np.array(
        [
            [0, 0, 0, 0.3, 0],
            [0, 0, 0, 0.2, 0],
            [0, 0, 0, 0.4, 0],
        ]
    )
    expected_t, expected_p = stats.ttest_rel(
        [0.8, 0.9, 0.7], [0.3, 0.2, 0.4]
    )

    result = compare_rep_route_crp(observed, control, max_lag=2, direction="same")

    assert isinstance(result, RepRouteCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert np.isclose(result.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(result.t_pvals[3], expected_p, atol=1e-6)
    assert result.direction == "same"
    np.testing.assert_array_equal(result.lags, np.arange(-2, 3))


def test_same_switch_bias_exact_difference():
    """Behavior: same-switch test compares observed bias against control bias."""
    observed_same = np.array(
        [
            [0, 0, 0, 0.8, 0],
            [0, 0, 0, 0.9, 0],
            [0, 0, 0, 0.7, 0],
        ]
    )
    observed_switch = np.array(
        [
            [0, 0, 0, 0.3, 0],
            [0, 0, 0, 0.2, 0],
            [0, 0, 0, 0.4, 0],
        ]
    )
    control_same = np.array(
        [
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0.4, 0],
        ]
    )
    control_switch = np.array(
        [
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, 0.4, 0],
        ]
    )

    result = compare_same_switch_bias(
        observed_same, observed_switch, control_same, control_switch, max_lag=2
    )

    assert isinstance(result, RepRouteCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert result.t_stats[3] > 0
    assert result.direction == "same-switch"


def test_plot_rep_route_crp_returns_axes():
    """Behavior: plot function returns a matplotlib Axes."""
    dataset = route_dataset()
    mask = jnp.ones(4, dtype=bool)
    axis = plot_rep_route_crp(
        dataset,
        mask,
        direction="same",
        use_lag2=False,
        min_lag=4,
        max_lag=3,
        labels=["Data"],
    )
    assert isinstance(axis, Axes)


def test_reproutecrp_jit_compatible():
    """Behavior: core route CRP is JIT compatible."""
    dataset = route_dataset()
    result_nojit = reproutecrp(dataset, direction="both", use_lag2=False, min_lag=4)
    result_jit = jit(
        reproutecrp, static_argnames=("direction", "use_lag2", "min_lag")
    )(dataset, direction="both", use_lag2=False, min_lag=4)

    np.testing.assert_allclose(
        np.array(result_nojit), np.array(result_jit), equal_nan=True
    )
