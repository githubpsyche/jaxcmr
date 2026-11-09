import numpy as np
import jax.numpy as jnp
from jaxcmr.analyses.log_odds_crp import log_odds_crp
from jaxcmr.helpers import make_dataset

LN2 = float(np.log(2))


def test_log_odds_crp_exact_logit_values():
    """Log-odds values match logit(CRP) minus reference."""
    trials = jnp.array(
        [[1, 2, 0, 0], [1, 2, 0, 0], [1, 3, 0, 0], [3, 2, 0, 0], [4, 2, 0, 0], [4, 3, 0, 0]],
        dtype=jnp.int32,
    )
    pres = jnp.tile(jnp.arange(1, 5, dtype=jnp.int32), (6, 1))
    dataset = make_dataset(trials, pres)
    lag_range = 3
    reference_lag = 2
    result = log_odds_crp(dataset, reference_lag=reference_lag, epsilon=1e-06, size=3)
    ref_idx = reference_lag + lag_range
    assert jnp.isclose(result[ref_idx], 0.0)
    assert jnp.isclose(result[1], 0.0)
    assert jnp.isclose(result[4], LN2, atol=0.0001)
    assert jnp.isclose(result[2], 2 * LN2, atol=0.0001)
    assert jnp.isnan(result[3])
