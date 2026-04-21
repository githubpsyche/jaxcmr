import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from scipy import stats

from jaxcmr.analyses.repcrp import (
    plot_first_rep_crp,
    plot_second_rep_crp,
    repcrp,
    test_rep_crp_vs_control as compare_rep_crp,
    test_first_second_bias as compare_first_second_bias,
    RepCRPTestResult,
)
from jaxcmr.helpers import make_dataset


def test_repcrp_exact_values_for_simple_repeated_list():
    """Behavior: CRP per presentation index matches hand-traced counts.

    Given:
      - 2 trials with presentation [1, 2, 1, 2] (items 1, 2 repeated).
      - Trial 1 recalls [1, 2]: lag +1 transitions from both presentations.
      - Trial 2 recalls [2, 1]: lag -1 transitions from both presentations.
      - min_lag=1 (spacing 2 >= 1 for both items).
    When:
      - ``repcrp`` is called with size=2.
    Then:
      - Both presentation indices show transitions at lags -1 and +1.
      - All observed lags have CRP 1.0 (every transition was taken).
    Why this matters:
      - Verifies that the repetition CRP correctly separates transitions
        by presentation index and produces exact conditional probabilities.
    """
    # Arrange / Given
    pres = jnp.array([[1, 2, 1, 2]] * 2, dtype=jnp.int32)
    recalls = jnp.array([[1, 2, 0, 0], [2, 1, 0, 0]], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres)

    # Act / When
    result = repcrp(dataset, min_lag=1, size=2)

    # Assert / Then — lags: [-3, -2, -1, 0, +1, +2, +3]
    assert result.shape == (2, 7)
    # Non-NaN values should all be 1.0 (every available transition was taken)
    finite = result[~jnp.isnan(result)]
    assert jnp.all(jnp.isclose(finite, 1.0))
    # Lag 0 (center, index 3) should be NaN for both presentations
    assert jnp.isnan(result[0, 3])
    assert jnp.isnan(result[1, 3])


def test_repcrp_values_are_valid_probabilities():
    """Behavior: All CRP values lie in [0, 1] or are NaN.

    Given:
      - A 4-trial repeated-item dataset.
    When:
      - ``repcrp`` is called.
    Then:
      - Every non-NaN value is in [0, 1].
    Why this matters:
      - CRP values are conditional probabilities and must be bounded.
    """
    # Arrange / Given
    pres = jnp.array([[1, 2, 3, 4, 5, 1, 2, 8]] * 4, dtype=jnp.int32)
    recalls = jnp.array([
        [1, 3, 2, 4, 0, 0, 0, 0],
        [1, 2, 3, 5, 0, 0, 0, 0],
        [3, 1, 2, 4, 5, 0, 0, 0],
        [1, 3, 5, 2, 0, 0, 0, 0],
    ], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres, subject=jnp.array([0, 0, 1, 1]))

    # Act / When
    result = repcrp(dataset, min_lag=4, size=2)

    # Assert / Then
    finite = result[~jnp.isnan(result)]
    assert jnp.all(finite >= 0.0).item()
    assert jnp.all(finite <= 1.0).item()


def test_rep_crp_vs_control_exact_t_stat():
    """Behavior: t-stat matches scipy ttest_rel for controlled inputs.

    Given:
      - 3 subjects: observed CRP at lag +1 = [0.8, 0.9, 0.7] (1st pres),
        control CRP at lag +1 = [0.3, 0.2, 0.4].
      - 2nd presentation has identical observed and control values.
    When:
      - ``test_rep_crp_vs_control`` is called.
    Then:
      - 1st presentation at lag +1: mean_diff = 0.5,
        t-stat matches scipy.stats.ttest_rel([0.8,0.9,0.7],[0.3,0.2,0.4]).
      - 2nd presentation at lag +1: mean_diff = 0.0 (no difference).
    Why this matters:
      - Verifies the exact paired t-test computation against a reference
        implementation, not just the presence of keys.
    """
    # Arrange / Given
    observed = np.array([
        [[0, 0, 0, 0.8, 0], [0, 0, 0, 0.3, 0]],
        [[0, 0, 0, 0.9, 0], [0, 0, 0, 0.2, 0]],
        [[0, 0, 0, 0.7, 0], [0, 0, 0, 0.4, 0]],
    ])
    control = np.array([
        [[0, 0, 0, 0.3, 0], [0, 0, 0, 0.3, 0]],
        [[0, 0, 0, 0.2, 0], [0, 0, 0, 0.2, 0]],
        [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
    ])
    expected_t, expected_p = stats.ttest_rel([0.8, 0.9, 0.7], [0.3, 0.2, 0.4])

    # Act / When
    results = compare_rep_crp(observed, control, max_lag=2)

    # Assert / Then
    first = results["First Presentation"]
    assert isinstance(first, RepCRPTestResult)
    assert np.isclose(first.mean_diffs[3], 0.5, atol=1e-10)
    assert np.isclose(first.t_stats[3], expected_t, atol=1e-6)
    assert np.isclose(first.t_pvals[3], expected_p, atol=1e-6)

    second = results["Second Presentation"]
    assert np.isclose(second.mean_diffs[3], 0.0, atol=1e-10)


def test_first_second_bias_positive_when_first_pres_dominates():
    """Behavior: Positive bias when 1st presentation CRP exceeds 2nd.

    Given:
      - 3 subjects where observed 1st-presentation CRP at lag +1
        exceeds 2nd-presentation by a constant 0.5.
      - Control has equal 1st and 2nd presentation CRP.
    When:
      - ``test_first_second_bias`` is called.
    Then:
      - Mean difference at lag +1 = 0.5 (1st pres advantage).
      - t-stat at lag +1 is large and positive.
    Why this matters:
      - Verifies the bias test correctly detects when first-presentation
        transitions dominate second-presentation transitions.
    """
    # Arrange / Given
    observed = np.array([
        [[0, 0, 0, 0.8, 0], [0, 0, 0, 0.3, 0]],
        [[0, 0, 0, 0.9, 0], [0, 0, 0, 0.2, 0]],
        [[0, 0, 0, 0.7, 0], [0, 0, 0, 0.4, 0]],
    ])
    control = np.array([
        [[0, 0, 0, 0.3, 0], [0, 0, 0, 0.3, 0]],
        [[0, 0, 0, 0.2, 0], [0, 0, 0, 0.2, 0]],
        [[0, 0, 0, 0.4, 0], [0, 0, 0, 0.4, 0]],
    ])

    # Act / When
    result = compare_first_second_bias(observed, control, max_lag=2)

    # Assert / Then
    assert isinstance(result, RepCRPTestResult)
    assert np.isclose(result.mean_diffs[3], 0.5, atol=1e-6)
    assert result.t_stats[3] > 0  # positive bias toward 1st pres
    np.testing.assert_array_equal(result.lags, np.arange(-2, 3))


def test_rep_crp_wrappers_use_dataset_labels_for_parameter_sweeps():
    """Behavior: single-repetition CRP wrappers label multiple datasets.

    Given:
      - Two datasets passed to a single-repetition wrapper.
      - Labels that represent parameter values.
    When:
      - ``plot_first_rep_crp`` and ``plot_second_rep_crp`` are called.
    Then:
      - Legend labels use the dataset labels, not the repetition index.
    Why this matters:
      - Parameter-shift plots overlay one curve per parameter value, so labels
        must identify sweep values rather than all saying ``1`` or ``2``.
    """
    pres = jnp.array([[1, 2, 3, 4, 5, 1, 2, 8]] * 4, dtype=jnp.int32)
    recalls = jnp.array([
        [1, 3, 2, 4, 0, 0, 0, 0],
        [1, 2, 3, 5, 0, 0, 0, 0],
        [3, 1, 2, 4, 5, 0, 0, 0],
        [1, 3, 5, 2, 0, 0, 0, 0],
    ], dtype=jnp.int32)
    dataset = make_dataset(recalls, pres, subject=jnp.array([0, 0, 1, 1]))
    mask = jnp.ones(4, dtype=bool)

    first_axis = plot_first_rep_crp(
        [dataset, dataset],
        [mask, mask],
        max_lag=3,
        min_lag=4,
        size=2,
        labels=["0.0", "1.0"],
    )
    first_labels = first_axis.get_legend_handles_labels()[1]
    assert "0.0" in first_labels
    assert "1.0" in first_labels

    second_axis = plot_second_rep_crp(
        [dataset, dataset],
        [mask, mask],
        max_lag=3,
        min_lag=4,
        size=2,
        labels=["0.0", "1.0"],
    )
    second_labels = second_axis.get_legend_handles_labels()[1]
    assert "0.0" in second_labels
    assert "1.0" in second_labels
