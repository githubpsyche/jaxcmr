import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit
from scipy import stats

from jaxcmr.analyses.repneighborcrp import (
    repneighborcrp,
    test_rep_neighbor_crp_vs_control as compare_rep_neighbor_crp,
    RepNeighborCRPTestResult,
    tabulate_trial,
)
from jaxcmr.helpers import make_dataset


_counts = jit(tabulate_trial, static_argnames=("direction", "use_lag2", "min_lag"))
_presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 3, 10, 11, 12])


def test_padding_does_not_invent_a_transition_or_opportunity():
    for recalls in ([4, 0, 0], [0, 0, 0], [0, 4, 0]):
        for direction in ("i2j", "j2i", "both"):
            actual, possible = _counts(jnp.array(recalls), _presentation,
                                      direction=direction, use_lag2=True, min_lag=4)
            assert not np.any(actual)
            assert not np.any(possible)


def test_real_last_item_is_still_counted():
    actual, possible = _counts(jnp.array([4, 12, 0]), _presentation,
                              direction="i2j", use_lag2=True, min_lag=4)
    # 4 is first-occurrence +1; 12 is second-occurrence +3.
    assert np.asarray(actual).sum() == 1
    assert actual[11 + 3] == possible[11 + 3] == 1


def test_last_position_is_not_aliased_to_the_penultimate_cue():
    actual, possible = _counts(jnp.array([12, 4]), _presentation,
                              direction="j2i", use_lag2=True, min_lag=4)
    # Position 12 is +3 from the second occurrence at 9, outside +1/+2.
    assert not np.any(actual)
    assert not np.any(possible)


def test_valid_last_position_neighbor_is_not_dropped():
    presentation = jnp.arange(1, 13).at[10].set(3)
    actual, possible = _counts(jnp.array([12, 4]), presentation,
                              direction="j2i", use_lag2=False, min_lag=4)
    # Position 12 is now +1 from the second occurrence at 11.
    assert np.asarray(actual).sum() == 1
    assert actual[11 + 1] == possible[11 + 1] == 1


@pytest.mark.parametrize("direction", ["i2j", "j2i", "both"])
@pytest.mark.parametrize("use_lag2", [False, True])
def test_inserting_zeros_preserves_transitions_and_availability(direction, use_lag2):
    for sequence in ([4, 12], [4, 10, 3, 12], [10, 4, 12], [2, 4, 12, 3]):
        expected = _counts(jnp.array(sequence), _presentation,
                           direction=direction, use_lag2=use_lag2, min_lag=4)
        padded = [0]
        for value in sequence:
            padded.extend([value, 0, 0])
        observed = _counts(jnp.array(padded), _presentation,
                           direction=direction, use_lag2=use_lag2, min_lag=4)
        for correct, result in zip(expected, observed):
            np.testing.assert_array_equal(result, correct)


def test_padding_does_not_consume_a_repeated_final_item():
    presentation = _presentation.at[11].set(4)
    expected = _counts(jnp.array([10, 4, 3]), presentation,
                       direction="both", use_lag2=True, min_lag=4)
    actual = _counts(jnp.array([0, 10, 0, 4, 0, 3, 0]), presentation,
                     direction="both", use_lag2=True, min_lag=4)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def test_repneighborcrp_exact_values():
    """Behavior: Neighbor CRP values match hand-traced lag counts.

    Given:
      - 2 trials with presentation [1, 2, 1, 2] and recalls
        [1, 2] and [2, 1].
      - direction="both", min_lag=1 (spacing 2 >= 1).
    When:
      - ``repneighborcrp`` is called.
    Then:
      - Shape is (7,) for 4-item list (2*4-1).
      - Non-NaN values are all 1.0 (every available neighbor
        transition was taken).
    Why this matters:
      - Verifies that the neighbor CRP correctly identifies transitions
        to neighbors of repeated items and computes exact probabilities.
    """
    # Arrange / Given
    pres = jnp.array([[1, 2, 1, 2]] * 2, dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 0, 0], [2, 1, 0, 0]], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres)

    # Act / When
    result = repneighborcrp(dataset, direction="both", use_lag2=False, min_lag=1)

    # Assert / Then
    assert result.shape == (7,)
    finite = result[~jnp.isnan(result)]
    assert jnp.all(jnp.isclose(finite, 1.0))


def test_rep_neighbor_crp_vs_control_exact_t_stat():
    """Behavior: t-stat matches scipy ttest_rel for controlled inputs.

    Given:
      - 3 subjects: observed neighbor CRP at lag +1 = [0.8, 0.9, 0.7],
        control = [0.3, 0.2, 0.4].
    When:
      - ``test_rep_neighbor_crp_vs_control`` is called.
    Then:
      - mean_diff at lag +1 = 0.5.
      - t-stat matches scipy.stats.ttest_rel exactly.
      - direction = "both".
    Why this matters:
      - Verifies the paired t-test computation and that the direction
        attribute is correctly propagated from input.
    """
    # Arrange / Given
    observed = np.array([
        [0, 0, 0, 0.8, 0],
        [0, 0, 0, 0.9, 0],
        [0, 0, 0, 0.7, 0],
    ])
    control = np.array([
        [0, 0, 0, 0.3, 0],
        [0, 0, 0, 0.2, 0],
        [0, 0, 0, 0.4, 0],
    ])
    expected_t, expected_p = stats.ttest_rel(
        [0.8, 0.9, 0.7], [0.3, 0.2, 0.4]
    )

    # Act / When
    result = compare_rep_neighbor_crp(
        observed, control, max_lag=2, direction="both"
    )

    # Assert / Then
    assert isinstance(result, RepNeighborCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert np.isclose(result.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(result.t_pvals[3], expected_p, atol=1e-6)
    assert result.direction == "both"
    np.testing.assert_array_equal(result.lags, np.arange(-2, 3))
