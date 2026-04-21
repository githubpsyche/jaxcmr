import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from jax import jit
from matplotlib.axes import Axes
from scipy import stats

from jaxcmr.analyses.repaccesscrp import (
    RepAccessCRPTestResult,
    plot_rep_access_crp,
    repaccesscrp,
    subject_rep_access_crp,
    tabulate_trial,
    test_first_second_bias as compare_first_second_bias,
    test_rep_access_crp_vs_control as compare_rep_access_crp,
)
from jaxcmr.helpers import make_dataset


def access_dataset():
    """Small dataset with first- and second-neighbor access transitions."""
    pres = jnp.array([[1, 2, 3, 4, 5, 1, 6, 7]] * 4, dtype=jnp.int32)
    recalls = jnp.array(
        [
            [2, 1, 0, 0, 0, 0, 0, 0],
            [7, 1, 0, 0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0, 0, 0, 0],
            [8, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    subjects = jnp.array([0, 0, 1, 1])
    return make_dataset(recalls, pres, subject=subjects)


def test_tabulate_trial_counts_i_neighbor_to_repeater():
    """Behavior: tabulation counts i+1 -> R access.

    Given:
      - Item 1 appears at positions i=1 and j=6.
      - Recall order [2, 1] is i+1 -> R.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - First-presentation lag +1 and second-presentation lag -4 are
        counted as both available and actual access to R.
    Why this matters:
      - Verifies the forward cue-conditioned numerator and denominator.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert actual.shape == (2, 15)
    assert possible.shape == (2, 15)
    assert int(actual[0, lag_range + 1]) == 1
    assert int(possible[0, lag_range + 1]) == 1
    assert int(actual[1, lag_range - 4]) == 1
    assert int(possible[1, lag_range - 4]) == 1
    assert int(actual.sum()) == 2
    assert int(possible.sum()) == 2


def test_tabulate_trial_counts_j_neighbor_to_repeater():
    """Behavior: tabulation counts j+1 -> R access.

    Given:
      - Item 1 appears at positions i=1 and j=6.
      - Recall order [7, 1] is j+1 -> R.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - First-presentation lag +6 and second-presentation lag +1 are
        counted as both available and actual access to R.
    Why this matters:
      - Verifies that the second-presentation neighbor case is symmetric
        in implementation, even if empirically asymmetric.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([7, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert int(actual[0, lag_range + 6]) == 1
    assert int(possible[0, lag_range + 6]) == 1
    assert int(actual[1, lag_range + 1]) == 1
    assert int(possible[1, lag_range + 1]) == 1
    assert int(actual.sum()) == 2
    assert int(possible.sum()) == 2


def test_denominator_excludes_unavailable_repeater():
    """Behavior: opportunities require the repeated target to be available.

    Given:
      - Item 1 is recalled first, making both of its study positions
        unavailable.
      - The next transition starts from R and goes to item 2.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - No opportunities to access R are counted.
    Why this matters:
      - Verifies repeated-recall guarding and prevents already-recalled
        repeaters from inflating denominators.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert int(actual.sum()) == 0
    assert int(possible.sum()) == 0


def test_tabulate_trial_ignores_leading_zero_sentinels():
    """Behavior: leading ``0`` sentinels do not seed previous context."""
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    compact = jnp.array([2, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    padded = jnp.array([0, 2, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)

    compact_actual, compact_possible = tabulate_trial(
        compact, pres, min_lag=4, size=2
    )
    padded_actual, padded_possible = tabulate_trial(padded, pres, min_lag=4, size=2)

    np.testing.assert_array_equal(np.array(compact_actual), np.array(padded_actual))
    np.testing.assert_array_equal(np.array(compact_possible), np.array(padded_possible))


def test_tabulate_trial_all_zero_returns_empty_counts():
    """Behavior: all-zero recall rows produce no access counts."""
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.zeros(8, dtype=jnp.int32)

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert actual.shape == (2, 15)
    assert possible.shape == (2, 15)
    assert int(actual.sum()) == 0
    assert int(possible.sum()) == 0


def test_repeated_previous_item_counts_all_study_positions():
    """Behavior: repeated previous items contribute all study positions.

    Given:
      - Item 1 is studied twice, at positions 1 and 3.
      - Item 2 is also studied twice, at positions 2 and 4.
      - The recall transition is [1, 2], so item 1 is the previous item
        and item 2 is the repeated target.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - The repeated target's denominator includes both previous study
        positions of item 1, producing two opportunities at lag -1.
      - The other lags from the same transition are still counted
        separately.
    Why this matters:
      - Catches regressions where multiple previous study positions are
        collapsed into a single binary opportunity.
    """
    pres = jnp.array([1, 2, 1, 2], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=0, size=2)

    assert actual.shape == (2, 7)
    assert possible.shape == (2, 7)
    assert int(actual[0, lag_range - 1]) == 1
    assert int(possible[0, lag_range - 1]) == 1
    assert int(actual[0, lag_range + 1]) == 1
    assert int(possible[0, lag_range + 1]) == 1
    assert int(actual[1, lag_range - 3]) == 1
    assert int(possible[1, lag_range - 3]) == 1
    assert int(actual[1, lag_range - 1]) == 1
    assert int(possible[1, lag_range - 1]) == 1
    assert int(actual.sum()) == 4
    assert int(possible.sum()) == 4


def test_repaccesscrp_shapes_and_probability_range():
    """Behavior: access CRP returns expected shape and valid probabilities."""
    dataset = access_dataset()

    result = repaccesscrp(dataset, min_lag=4, size=2)

    assert result.shape == (2, 15)
    finite = result[~jnp.isnan(result)]
    assert jnp.all(finite >= 0.0).item()
    assert jnp.all(finite <= 1.0).item()


def test_subject_rep_access_crp_shape():
    """Behavior: subject-level access CRP preserves subject and index axes."""
    dataset = access_dataset()
    mask = jnp.ones(4, dtype=bool)

    result = subject_rep_access_crp(dataset, mask, min_lag=4, max_lag=3, size=2)

    assert result.shape == (2, 2, 7)


def test_rep_access_crp_vs_control_exact_t_stat():
    """Behavior: t-stat matches scipy ttest_rel for known inputs."""
    observed = np.array([
        [[0, 0, 0, 0.8, 0], [0, 0, 0, 0.2, 0]],
        [[0, 0, 0, 0.9, 0], [0, 0, 0, 0.1, 0]],
        [[0, 0, 0, 0.7, 0], [0, 0, 0, 0.3, 0]],
    ])
    control = np.array([
        [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
        [[0, 0, 0, 0.3, 0], [0, 0, 0, 0.3, 0]],
        [[0, 0, 0, 0.5, 0], [0, 0, 0, 0.5, 0]],
    ])
    expected_t, expected_p = stats.ttest_rel(
        [0.8, 0.9, 0.7], [0.4, 0.3, 0.5]
    )

    results = compare_rep_access_crp(observed, control, max_lag=2)

    first = results["First Presentation"]
    assert isinstance(first, RepAccessCRPTestResult)
    assert np.isclose(first.mean_diffs[3], 0.4, atol=1e-6)
    assert np.isclose(first.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[3], expected_p, atol=1e-6)


def test_first_second_bias_exact_difference():
    """Behavior: first-second test compares observed bias against control bias."""
    observed = np.array(
        [
            [[0, 0, 0, 0.8, 0], [0, 0, 0, 0.3, 0]],
            [[0, 0, 0, 0.9, 0], [0, 0, 0, 0.2, 0]],
            [[0, 0, 0, 0.7, 0], [0, 0, 0, 0.4, 0]],
        ]
    )
    control = np.array(
        [
            [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
            [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
            [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
        ]
    )

    result = compare_first_second_bias(observed, control, max_lag=2)

    assert isinstance(result, RepAccessCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert result.t_stats[3] > 0


def test_plot_rep_access_crp_returns_axes():
    """Behavior: plot function returns a matplotlib Axes."""
    dataset = access_dataset()
    mask = jnp.ones(4, dtype=bool)

    axis = plot_rep_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        size=2,
        labels=["First", "Second"],
    )

    assert isinstance(axis, Axes)


def test_repaccesscrp_jit_compatible():
    """Behavior: core access CRP is JIT compatible."""
    dataset = access_dataset()
    result_nojit = repaccesscrp(dataset, min_lag=4, size=2)
    result_jit = jit(repaccesscrp, static_argnames=("min_lag", "size"))(
        dataset, min_lag=4, size=2
    )

    np.testing.assert_allclose(
        np.array(result_nojit), np.array(result_jit), equal_nan=True
    )
