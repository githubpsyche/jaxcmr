import numpy as np
import jax.numpy as jnp

from jaxcmr.analyses.log_odds_crp import simple_crp, log_odds_crp
from jaxcmr.helpers import make_dataset

LN2 = float(np.log(2))


def test_simple_crp_exact_values():
    """Behavior: CRP matches hand-calculated transition probabilities.

    Given:
      - Six 4-item trials with known single transitions:
        trials 1-2 at lag +1, trial 3 at lag +2, trials 4,6 at lag -1,
        trial 5 at lag -2.
    When:
      - ``simple_crp`` is called.
    Then:
      - CRP = [0, 1/3, 2/3, NaN, 1/2, 1/3, 0] for lags [-3..+3].
    Why this matters:
      - Verifies exact conditional probabilities computed from
        aggregated actual / available lag counts across trials.
    """
    # Arrange / Given
    trials = jnp.array([
        [1, 2, 0, 0],  # lag +1
        [1, 2, 0, 0],  # lag +1
        [1, 3, 0, 0],  # lag +2
        [3, 2, 0, 0],  # lag -1
        [4, 2, 0, 0],  # lag -2
        [4, 3, 0, 0],  # lag -1
    ], dtype=jnp.int32)

    # Act / When
    result = simple_crp(trials, 4)

    # Assert / Then — lags: [-3, -2, -1, 0, +1, +2, +3]
    assert jnp.isclose(result[0], 0.0)            # lag -3: 0/2
    assert jnp.isclose(result[1], 1.0 / 3.0)      # lag -2: 1/3
    assert jnp.isclose(result[2], 2.0 / 3.0)      # lag -1: 2/3
    assert jnp.isnan(result[3])                    # lag  0: 0/0
    assert jnp.isclose(result[4], 0.5)             # lag +1: 2/4
    assert jnp.isclose(result[5], 1.0 / 3.0)      # lag +2: 1/3
    assert jnp.isclose(result[6], 0.0)             # lag +3: 0/3


def test_log_odds_crp_exact_logit_values():
    """Behavior: Log-odds values match logit(CRP) minus reference.

    Given:
      - The same 6-trial dataset producing CRP [0,1/3,2/3,NaN,1/2,1/3,0].
      - Reference lag +2 (CRP = 1/3 at that lag).
    When:
      - ``log_odds_crp`` is called.
    Then:
      - Log-odds at lag +2 (reference) = 0.0.
      - Log-odds at lag -2 = 0.0 (same CRP 1/3 as reference).
      - Log-odds at lag +1 = ln(2) (logit(1/2) - logit(1/3) = ln(2)).
      - Log-odds at lag -1 = 2*ln(2) (logit(2/3) - logit(1/3) = 2*ln(2)).
    Why this matters:
      - Verifies the exact logit transform and reference subtraction for
        lags with fractional CRP values (no clipping artifacts).
    """
    # Arrange / Given
    trials = jnp.array([
        [1, 2, 0, 0], [1, 2, 0, 0], [1, 3, 0, 0],
        [3, 2, 0, 0], [4, 2, 0, 0], [4, 3, 0, 0],
    ], dtype=jnp.int32)
    pres = jnp.tile(jnp.arange(1, 5, dtype=jnp.int32), (6, 1))
    dataset = make_dataset(trials, pres)
    lag_range = 3
    reference_lag = 2

    # Act / When
    result = log_odds_crp(dataset, reference_lag=reference_lag, epsilon=1e-6, size=3)

    # Assert / Then
    ref_idx = reference_lag + lag_range  # index 5
    assert jnp.isclose(result[ref_idx], 0.0)                      # reference = 0.0
    assert jnp.isclose(result[1], 0.0)                            # lag -2 = 0.0 (same CRP)
    assert jnp.isclose(result[4], LN2, atol=1e-4)                 # lag +1 = ln(2)
    assert jnp.isclose(result[2], 2 * LN2, atol=1e-4)            # lag -1 = 2*ln(2)
    assert jnp.isnan(result[3])                                    # lag  0: NaN


def test_log_odds_crp_ordering_matches_crp_ordering():
    """Behavior: Higher CRP produces higher log-odds.

    Given:
      - The same 6-trial dataset with CRP ordering: lag -1 > lag +1 > lag -2.
    When:
      - ``log_odds_crp`` is called.
    Then:
      - log_odds[lag -1] > log_odds[lag +1] > log_odds[lag -2].
    Why this matters:
      - The logit is a monotonic transform, so CRP ordering must be
        preserved in log-odds space.
    """
    # Arrange / Given
    trials = jnp.array([
        [1, 2, 0, 0], [1, 2, 0, 0], [1, 3, 0, 0],
        [3, 2, 0, 0], [4, 2, 0, 0], [4, 3, 0, 0],
    ], dtype=jnp.int32)
    pres = jnp.tile(jnp.arange(1, 5, dtype=jnp.int32), (6, 1))
    dataset = make_dataset(trials, pres)

    # Act / When
    result = log_odds_crp(dataset, reference_lag=2, epsilon=1e-6, size=3)

    # Assert / Then
    assert result[2] > result[4] > result[1]  # lag -1 > lag +1 > lag -2
