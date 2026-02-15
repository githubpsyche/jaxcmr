import jax.numpy as jnp
import numpy as np
from scipy import stats

from jaxcmr.analyses.cleanbackrepcrp import (
    subject_back_rep_crp,
    test_back_rep_crp_vs_control as compare_back_rep_crp,
    BackRepCRPTestResult,
)
from jaxcmr.helpers import make_dataset


def test_back_rep_crp_matches_forward_for_symmetric_recalls():
    """Behavior: Backward CRP equals forward for palindromic recall order.

    Given:
      - 2 trials with presentation [1, 2, 1, 2] and recalls [1, 2]
        and [2, 1]. Reversed: [2, 1] and [1, 2] — same set of sequences.
    When:
      - ``subject_back_rep_crp`` is called.
    Then:
      - Non-NaN values are all 1.0 (both transitions taken).
      - Shape is (1, 2, 7) for 1 subject, 2 presentation indices, 7 lags.
    Why this matters:
      - Verifies that the backward CRP (reversing recall order) correctly
        processes transitions and produces valid probabilities.
    """
    # Arrange / Given
    pres = jnp.array([[1, 2, 1, 2]] * 2, dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 0, 0], [2, 1, 0, 0]], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres)
    trial_mask = jnp.ones(2, dtype=bool)

    # Act / When
    result = subject_back_rep_crp(dataset, trial_mask, min_lag=1, max_lag=3, size=2)

    # Assert / Then
    assert result.shape == (1, 2, 7)
    finite = result[~np.isnan(np.array(result))]
    assert np.all(np.isclose(finite, 1.0))


def test_back_rep_crp_vs_control_exact_t_stat():
    """Behavior: t-stat matches scipy ttest_rel for known inputs.

    Given:
      - 3 subjects: observed 1st-pres backward CRP at lag +1 = [0.8, 0.9, 0.7],
        control = [0.4, 0.3, 0.5].
    When:
      - ``test_back_rep_crp_vs_control`` is called.
    Then:
      - 1st presentation at lag +1: mean_diff = 0.4,
        t-stat matches scipy.stats.ttest_rel.
    Why this matters:
      - Verifies the paired t-test computation for backward CRP against
        a reference implementation.
    """
    # Arrange / Given
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

    # Act / When
    results = compare_back_rep_crp(observed, control, max_lag=2)

    # Assert / Then
    first = results["First Presentation"]
    assert isinstance(first, BackRepCRPTestResult)
    assert np.isclose(first.mean_diffs[3], 0.4, atol=1e-6)
    assert np.isclose(first.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[3], expected_p, atol=1e-6)
