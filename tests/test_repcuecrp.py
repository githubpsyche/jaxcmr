import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from jax import jit
from matplotlib.axes import Axes
from scipy import stats

from jaxcmr.analyses.repcuecrp import (
    RepCueCRPTestResult,
    _access_band,
    _access_diagonal,
    _previous_next_surface,
    _repcuecrp_window,
    _subject_rep_cue_crp_full,
    plot_first_rep_cue_access_crp,
    plot_rep_cue_access_crp,
    plot_rep_cue_crp,
    plot_rep_cue_crp_surface,
    plot_rep_cue_offset_surface,
    plot_second_rep_cue_access_crp,
    repcuecrp,
    subject_rep_cue_crp,
    tabulate_trial,
    test_first_second_access_band_bias as compare_first_second_access_band_bias,
    test_first_second_access_bias as compare_first_second_access_bias,
    test_first_second_bias as compare_first_second_bias,
    test_rep_cue_access_band_vs_control as compare_rep_cue_access_band,
    test_rep_cue_access_vs_control as compare_rep_cue_access,
    test_rep_cue_crp_vs_control as compare_rep_cue_crp,
)
from jaxcmr.analyses.repaccesscrp import subject_rep_access_crp
from jaxcmr.helpers import make_dataset


def cue_dataset():
    """Small dataset with first- and second-neighbor cue transitions."""
    pres = jnp.array([[1, 2, 3, 4, 5, 1, 6, 7]] * 4, dtype=jnp.int32)
    recalls = jnp.array(
        [
            [2, 1, 0, 0, 0, 0, 0, 0],
            [7, 1, 0, 0, 0, 0, 0, 0],
            [2, 3, 0, 0, 0, 0, 0, 0],
            [7, 6, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    subjects = jnp.array([0, 0, 1, 1])
    return make_dataset(recalls, pres, subject=subjects)


def test_tabulate_trial_counts_i_neighbor_access_diagonal():
    """Behavior: tabulation counts i+1 -> R on the access diagonal.

    Given:
      - Item 1 appears at positions i=1 and j=6.
      - Recall order [2, 1] is i+1 -> R.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - Cell (first, cue offset +1, transition lag -1) is actual and possible.
      - The same recalled repeated item also contributes the lag to its
        other study position, preserving ordinary CRP ambiguity handling.
    Why this matters:
      - Verifies the diagonal access case without collapsing away the
        transition-lag axis.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert actual.shape == (2, 15, 15)
    assert possible.shape == (2, 15, 15)
    assert int(actual[0, lag_range + 1, lag_range - 1]) == 1
    assert int(possible[0, lag_range + 1, lag_range - 1]) == 1
    assert int(actual[0, lag_range + 1, lag_range + 4]) == 1
    assert int(possible[0, lag_range + 1, lag_range + 4]) == 1


def test_tabulate_trial_counts_j_neighbor_access_diagonal():
    """Behavior: tabulation counts j+1 -> R on the access diagonal.

    Given:
      - Item 1 appears at positions i=1 and j=6.
      - Recall order [7, 1] is j+1 -> R.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - Cell (second, cue offset +1, transition lag -1) is actual and possible.
    Why this matters:
      - Verifies that the second-presentation neighbor case is symmetric
        in implementation even if the empirical curve is asymmetric.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([7, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert int(actual[1, lag_range + 1, lag_range - 1]) == 1
    assert int(possible[1, lag_range + 1, lag_range - 1]) == 1
    assert int(actual[1, lag_range + 1, lag_range - 6]) == 1
    assert int(possible[1, lag_range + 1, lag_range - 6]) == 1


def test_non_access_transition_counts_ordinary_transition_lag():
    """Behavior: non-access transitions remain in the offset-specific CRP.

    Given:
      - Item 1 is repeated at positions 1 and 6.
      - Recall order [2, 3] starts from i+1 but does not recall R.
    When:
      - ``tabulate_trial`` is called.
    Then:
      - The first-presentation cue offset +1 row records transition lag +1.
      - The access diagonal lag -1 is possible but not actual.
    Why this matters:
      - Confirms this analysis is a prior-conditioned forward CRP, not
        only a target-access hazard.
    """
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([2, 3, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert int(actual[0, lag_range + 1, lag_range + 1]) == 1
    assert int(possible[0, lag_range + 1, lag_range + 1]) == 1
    assert int(actual[0, lag_range + 1, lag_range - 1]) == 0
    assert int(possible[0, lag_range + 1, lag_range - 1]) == 1


def test_cue_offset_zero_is_not_tabulated():
    """Behavior: transitions after the repeater itself skip cue offset 0."""
    pres = jnp.array([1, 2, 3, 4, 5, 1, 6, 7], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=4, size=2)

    assert int(actual[:, lag_range, :].sum()) == 0
    assert int(possible[:, lag_range, :].sum()) == 0


def test_ambiguous_positions_contribute_like_repcrp():
    """Behavior: repeated previous/current items contribute all positions.

    Given:
      - Items 1 and 2 are both repeated in [1, 2, 1, 2].
      - Recall order [1, 2] has a repeated previous item and a repeated
        current item.
    When:
      - ``tabulate_trial`` is called with min_lag=0.
    Then:
      - Both previous positions of item 1 and both current positions of
        item 2 contribute applicable offset/lag cells.
    Why this matters:
      - Matches ``repcrp`` ambiguity handling instead of excluding
        multiply mapped recalls.
    """
    pres = jnp.array([1, 2, 1, 2], dtype=jnp.int32)
    trial = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
    lag_range = pres.size - 1

    actual, possible = tabulate_trial(trial, pres, min_lag=0, size=2)

    assert int(actual[0, lag_range - 1, lag_range + 1]) == 1
    assert int(actual[0, lag_range - 1, lag_range + 3]) == 1
    assert int(actual[0, lag_range + 1, lag_range - 1]) == 1
    assert int(actual[0, lag_range + 1, lag_range + 1]) == 1
    assert int(possible[0, lag_range - 1, lag_range + 1]) == 1
    assert int(possible[0, lag_range + 1, lag_range - 1]) == 1


def test_repcuecrp_shapes_and_probability_range():
    """Behavior: cue CRP returns expected shape and valid probabilities."""
    dataset = cue_dataset()

    result = repcuecrp(dataset, min_lag=4, size=2)

    assert result.shape == (2, 15, 15)
    finite = result[~jnp.isnan(result)]
    assert jnp.all(finite >= 0.0).item()
    assert jnp.all(finite <= 1.0).item()


def test_subject_rep_cue_crp_shape():
    """Behavior: subject-level cue CRP preserves subject and cue axes."""
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    result = subject_rep_cue_crp(
        dataset, mask, min_lag=4, max_lag=3, max_offset=3, size=2
    )

    assert result.shape == (2, 2, 7, 7)


def test_repcuecrp_window_matches_full_slice():
    """Behavior: windowed cue CRP is equivalent to slicing full cue CRP."""
    dataset = cue_dataset()
    max_lag = 3
    max_offset = 3
    lag_range = dataset["pres_itemnos"].shape[1] - 1
    offset_slice = slice(lag_range - max_offset, lag_range + max_offset + 1)
    lag_slice = slice(lag_range - max_lag, lag_range + max_lag + 1)

    full = repcuecrp(dataset, min_lag=4, size=2)
    window = _repcuecrp_window(
        dataset, min_lag=4, max_lag=max_lag, max_offset=max_offset, size=2
    )

    np.testing.assert_allclose(
        np.array(full[:, offset_slice, lag_slice]),
        np.array(window),
        equal_nan=True,
    )


def test_subject_rep_cue_crp_matches_full_slice():
    """Behavior: subject-level optimized path matches full tabulation."""
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    full = _subject_rep_cue_crp_full(
        dataset, mask, min_lag=4, max_lag=3, max_offset=3, size=2
    )
    window = subject_rep_cue_crp(
        dataset, mask, min_lag=4, max_lag=3, max_offset=3, size=2
    )

    np.testing.assert_allclose(full, window, equal_nan=True)


def test_subject_rep_cue_crp_matches_jax_window_multi_subject_ambiguous():
    """Behavior: optimized subject path matches JAX window tabulation."""
    pres = jnp.array(
        [
            [1, 2, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 1, 6, 7],
            [1, 2, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 1, 6, 7],
        ],
        dtype=jnp.int32,
    )
    recalls = jnp.array(
        [
            [1, 2, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [7, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    subjects = jnp.array([0, 0, 1, 1])
    dataset = make_dataset(recalls, pres, subject=subjects)
    mask = jnp.ones(4, dtype=bool)

    result = subject_rep_cue_crp(
        dataset, mask, min_lag=0, max_lag=3, max_offset=3, size=2
    )
    subject_indices = np.asarray(dataset["subject"]).reshape(-1)
    expected = []
    for subject in np.unique(subject_indices):
        subject_mask = subject_indices == subject
        subject_dataset = {
            key: value[subject_mask]
            for key, value in dataset.items()
        }
        expected.append(
            _repcuecrp_window(
                subject_dataset, min_lag=0, max_lag=3, max_offset=3, size=2
            )
        )
    expected = np.stack(expected)

    np.testing.assert_allclose(expected, result, equal_nan=True)


def test_access_diagonal_extraction():
    """Behavior: access diagonal extracts transition lag = -cue offset.

    Offset 0 remains in the labels but is all NaN so plotted lines break
    at the repeated item.

    """
    crp = np.full((2, 2, 5, 5), np.nan)
    crp[:, 0, 3, 1] = 0.8  # cue offset +1, transition lag -1
    crp[:, 1, 1, 3] = 0.4  # cue offset -1, transition lag +1

    offsets, access = _access_diagonal(crp, max_offset=2, max_lag=2)

    np.testing.assert_array_equal(offsets, np.array([-2, -1, 0, 1, 2]))
    assert access.shape == (2, 2, 5)
    assert np.isnan(access[:, :, 2]).all()
    assert np.allclose(access[:, 0, 3], 0.8)
    assert np.allclose(access[:, 1, 1], 0.4)


def test_access_band_averages_selected_offsets():
    """Behavior: access band averages diagonal values over chosen offsets."""
    crp = np.full((3, 2, 5, 5), np.nan)
    crp[:, 0, 1, 3] = [0.2, 0.4, 0.6]  # cue offset -1, transition lag +1
    crp[:, 0, 3, 1] = [0.8, 0.6, 0.4]  # cue offset +1, transition lag -1

    offsets, band = _access_band(
        crp, cue_offsets=(-1, 1), max_offset=2, max_lag=2
    )

    np.testing.assert_array_equal(offsets, np.array([-1, 1]))
    np.testing.assert_allclose(band[:, 0], np.array([0.5, 0.5, 0.5]))
    assert np.isnan(band[:, 1]).all()


def test_repcue_access_diagonal_matches_repaccesscrp_nonzero_offsets():
    """Behavior: cue access diagonal matches the older access CRP module.

    Given:
      - The same intended preprocessing/settings.
    When:
      - ``repaccesscrp`` and ``repcuecrp`` are both reduced to the
        access-to-repeater curve.
    Then:
      - Their subject-level values match at nonzero offsets.
    Why this matters:
      - Supports moving the access analysis into ``repcuecrp`` without
        changing the measured quantity.
    """
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    access = subject_rep_access_crp(dataset, mask, min_lag=4, max_lag=3, size=2)
    cue = subject_rep_cue_crp(
        dataset, mask, min_lag=4, max_lag=3, max_offset=3, size=2
    )
    offsets, cue_access = _access_diagonal(cue, max_offset=3, max_lag=3)
    nonzero = offsets != 0

    np.testing.assert_allclose(
        access[:, :, nonzero],
        cue_access[:, :, nonzero],
        equal_nan=True,
    )


def test_previous_next_surface_transforms_known_cell():
    """Behavior: cue/lag cells map to previous/next offset cells.

    Given:
      - A value at cue offset +1 and transition lag -2.
    When:
      - ``_previous_next_surface`` is called.
    Then:
      - The value appears at previous offset +1 and next offset -1.
    Why this matters:
      - Verifies the transformed heatmap changes coordinates, not values.
    """
    crp = np.full((1, 2, 5, 5), np.nan)
    crp[:, 0, 3, 0] = 0.8  # previous +1, transition -2 -> next -1

    offsets, transformed = _previous_next_surface(crp, max_offset=2, max_lag=2)

    np.testing.assert_array_equal(offsets, np.arange(-2, 3))
    assert transformed.shape == (1, 2, 5, 5)
    assert np.isclose(transformed[0, 0, 1, 3], 0.8)


def test_previous_next_surface_maps_access_to_zero_next_offset():
    """Behavior: access diagonal lands on next offset 0.

    Given:
      - Values on cells where transition lag = -cue offset.
    When:
      - ``_previous_next_surface`` is called.
    Then:
      - Both values appear in the next-offset 0 row.
    Why this matters:
      - The transformed heatmap should make repeater access visually fixed.
    """
    crp = np.full((1, 2, 5, 5), np.nan)
    crp[:, 0, 3, 1] = 0.8  # previous +1, transition -1 -> next 0
    crp[:, 1, 1, 3] = 0.4  # previous -1, transition +1 -> next 0

    _, transformed = _previous_next_surface(crp, max_offset=2, max_lag=2)

    assert np.isclose(transformed[0, 0, 2, 3], 0.8)
    assert np.isclose(transformed[0, 1, 2, 1], 0.4)


def test_previous_next_surface_leaves_out_of_range_cells_nan():
    """Behavior: transformed cells outside offset range remain NaN."""
    crp = np.full((1, 2, 5, 5), np.nan)
    crp[:, 0, 4, 4] = 0.8  # previous +2, transition +2 -> next +4

    _, transformed = _previous_next_surface(crp, max_offset=2, max_lag=2)

    assert np.isnan(transformed[0, 0]).all()


def test_rep_cue_crp_vs_control_exact_t_stat():
    """Behavior: offset-specific t-stat matches scipy ttest_rel."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 3, 3] = [0.8, 0.9, 0.7]
    control[:, 0, 3, 3] = [0.3, 0.2, 0.4]
    expected_t, expected_p = stats.ttest_rel(
        [0.8, 0.9, 0.7], [0.3, 0.2, 0.4]
    )

    results = compare_rep_cue_crp(
        observed, control, cue_offset=1, max_offset=2, max_lag=2
    )

    first = results["First Presentation"]
    assert isinstance(first, RepCueCRPTestResult)
    assert np.isclose(first.mean_diffs[3], 0.5, atol=1e-10)
    assert np.isclose(first.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[3], expected_p, atol=1e-6)


def test_rep_cue_access_vs_control_exact_t_stat():
    """Behavior: access-diagonal t-stat matches scipy ttest_rel."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 3, 1] = [0.8, 0.9, 0.7]
    control[:, 0, 3, 1] = [0.3, 0.2, 0.4]
    expected_t, expected_p = stats.ttest_rel(
        [0.8, 0.9, 0.7], [0.3, 0.2, 0.4]
    )

    results = compare_rep_cue_access(
        observed, control, max_offset=2, max_lag=2
    )

    first = results["First Presentation"]
    assert isinstance(first, RepCueCRPTestResult)
    assert np.isnan(first.mean_diffs[2])
    assert np.isclose(first.mean_diffs[3], 0.5, atol=1e-10)
    assert np.isclose(first.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[3], expected_p, atol=1e-6)
    assert first.label_name == "Offset"
    assert str(first).splitlines()[0].strip().startswith("Offset")


def test_rep_cue_access_band_vs_control_exact_t_stat():
    """Behavior: aggregate access-band t-stat matches scipy ttest_rel."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 1, 3] = [0.6, 0.8, 0.7]
    observed[:, 0, 3, 1] = [1.0, 1.0, 0.7]
    control[:, 0, 1, 3] = [0.2, 0.4, 0.3]
    control[:, 0, 3, 1] = [0.4, 0.4, 0.3]
    observed_band = [0.8, 0.9, 0.7]
    control_band = [0.3, 0.4, 0.3]
    expected_t, expected_p = stats.ttest_rel(observed_band, control_band)

    results = compare_rep_cue_access_band(
        observed, control, cue_offsets=(-1, 1), max_offset=2, max_lag=2
    )

    first = results["First Presentation"]
    assert isinstance(first, RepCueCRPTestResult)
    assert first.label_name == "Offsets"
    assert first.lags[0] == "-1,+1"
    assert np.isclose(first.mean_diffs[0], 0.4666666666666667, atol=1e-10)
    assert np.isclose(first.t_stats[0], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[0], expected_p, atol=1e-6)


def test_first_second_bias_exact_difference():
    """Behavior: first-second cue test compares bias against control bias."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 3, 3] = [0.8, 0.9, 0.7]
    observed[:, 1, 3, 3] = [0.3, 0.2, 0.4]
    control[:, 0, 3, 3] = [0.4, 0.4, 0.4]
    control[:, 1, 3, 3] = [0.4, 0.4, 0.4]

    result = compare_first_second_bias(
        observed, control, cue_offset=1, max_offset=2, max_lag=2
    )

    assert isinstance(result, RepCueCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert result.t_stats[3] > 0


def test_first_second_access_bias_exact_difference():
    """Behavior: first-second access test compares bias against control bias."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 3, 1] = [0.8, 0.9, 0.7]
    observed[:, 1, 3, 1] = [0.3, 0.2, 0.4]
    control[:, 0, 3, 1] = [0.4, 0.4, 0.4]
    control[:, 1, 3, 1] = [0.4, 0.4, 0.4]

    result = compare_first_second_access_bias(observed, control, 2, 2)

    assert isinstance(result, RepCueCRPTestResult)
    assert np.isnan(result.mean_diffs[2])
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert result.t_stats[3] > 0


def test_first_second_access_band_bias_exact_difference():
    """Behavior: aggregate first-second access bias compares to control."""
    observed = np.zeros((3, 2, 5, 5), dtype=float)
    control = np.zeros((3, 2, 5, 5), dtype=float)
    observed[:, 0, 1, 3] = [0.6, 0.8, 0.7]
    observed[:, 0, 3, 1] = [1.0, 1.0, 0.7]
    observed[:, 1, 1, 3] = [0.2, 0.4, 0.3]
    observed[:, 1, 3, 1] = [0.4, 0.4, 0.3]
    control[:, 0, 1, 3] = [0.4, 0.4, 0.4]
    control[:, 0, 3, 1] = [0.4, 0.4, 0.4]
    control[:, 1, 1, 3] = [0.4, 0.4, 0.4]
    control[:, 1, 3, 1] = [0.4, 0.4, 0.4]

    result = compare_first_second_access_band_bias(
        observed, control, cue_offsets=(-1, 1), max_offset=2, max_lag=2
    )

    assert isinstance(result, RepCueCRPTestResult)
    assert result.label_name == "Offsets"
    assert result.lags[0] == "-1,+1"
    assert np.isclose(result.mean_diffs[0], 0.4666666666666667, atol=1e-10)
    assert result.t_stats[0] > 0


def test_plot_wrappers_return_axes():
    """Behavior: cue plot wrappers return matplotlib Axes."""
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    axis = plot_rep_cue_crp(
        dataset,
        mask,
        cue_offset=1,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        labels=["First", "Second"],
    )
    assert isinstance(axis, Axes)

    axis = plot_rep_cue_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        labels=["First", "Second"],
    )
    assert isinstance(axis, Axes)

    axis = plot_rep_cue_crp_surface(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        repetition_index=0,
    )
    assert isinstance(axis, Axes)

    axis = plot_rep_cue_offset_surface(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        repetition_index=0,
    )
    assert isinstance(axis, Axes)


def test_access_wrapper_lines_match_repetition_index_argument():
    """Behavior: first/second wrappers match explicit repetition-index calls."""
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    def first_nonempty_ydata(axis):
        for line in axis.get_lines():
            ydata = np.asarray(line.get_ydata())
            if ydata.size:
                return ydata.astype(float)
        raise AssertionError("No non-empty plotted line found.")

    first_axis = plot_first_rep_cue_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
    )
    direct_first_axis = plot_rep_cue_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        repetition_index=0,
    )
    assert np.allclose(
        first_nonempty_ydata(first_axis),
        first_nonempty_ydata(direct_first_axis),
        equal_nan=True,
    )

    second_axis = plot_second_rep_cue_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
    )
    direct_second_axis = plot_rep_cue_access_crp(
        dataset,
        mask,
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        repetition_index=1,
    )
    assert np.allclose(
        first_nonempty_ydata(second_axis),
        first_nonempty_ydata(direct_second_axis),
        equal_nan=True,
    )


def test_access_wrapper_uses_dataset_labels_for_parameter_sweeps():
    """Behavior: single-repetition access wrappers label multiple datasets."""
    dataset = cue_dataset()
    mask = jnp.ones(4, dtype=bool)

    axis = plot_first_rep_cue_access_crp(
        [dataset, dataset],
        [mask, mask],
        min_lag=4,
        max_lag=3,
        max_offset=3,
        size=2,
        labels=["0.0", "1.0"],
    )
    labels = axis.get_legend_handles_labels()[1]
    assert "0.0" in labels
    assert "1.0" in labels


def test_repcuecrp_jit_compatible():
    """Behavior: core cue CRP is JIT compatible."""
    dataset = cue_dataset()
    result_nojit = repcuecrp(dataset, min_lag=4, size=2)
    result_jit = jit(repcuecrp, static_argnames=("min_lag", "size"))(
        dataset, min_lag=4, size=2
    )

    np.testing.assert_allclose(
        np.array(result_nojit), np.array(result_jit), equal_nan=True
    )
