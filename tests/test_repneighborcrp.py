import jax.numpy as jnp
import numpy as np
from scipy import stats

from jaxcmr.analyses.repneighborcrp import (
    repneighborcrp,
    test_rep_neighbor_crp_vs_control as compare_rep_neighbor_crp,
    RepNeighborCRPTestResult,
)
from jaxcmr.helpers import make_dataset


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
