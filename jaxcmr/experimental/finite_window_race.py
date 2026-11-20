"""Independent diffusion accumulators race to a common boundary at each recall.

A new unit-duration window starts at every retrieval attempt. STOP means
that all eligible accumulators survived to the deadline. Each item gets
the probability that its accumulator was the first to finish in that window.
"""

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.nn import softmax
from jax.scipy.special import log_ndtr, logsumexp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.response import SupportsItemEvidence
from jaxcmr.typing import Array, Bool, Float, Float_, Parameters

_nodes, _weights = np.polynomial.legendre.leggauss(64)
_NODES = (_nodes + 1) / 2
_WEIGHTS = _weights / 2


def _log_survival(
    boundary: Float_,
    drift: Float[Array, "..."],
    time: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    """Log survival of a unit-noise Brownian first-passage time."""
    root_time = jnp.sqrt(time)
    distance = (boundary - drift * time) / root_time
    reflected = 2 * boundary * drift + log_ndtr(-(boundary + drift * time) / root_time)
    log_cdf = jnp.logaddexp(log_ndtr(-distance), reflected)
    # Use the smaller tail to avoid subtracting nearly equal probabilities.
    left = log_ndtr(distance)
    log_half = -jnp.log(2.0)
    log_from_cdf = jnp.log1p(-jnp.exp(jnp.minimum(log_cdf, log_half)))
    difference = jnp.minimum(reflected - left, -jnp.finfo(left.dtype).eps)
    return lax.select(
        log_cdf < log_half, log_from_cdf, left + jnp.log(-jnp.expm1(difference))
    )


def finite_window_probabilities(
    drifts: Float[Array, " items"],
    eligible: Bool[Array, " items"],
    boundary: Float_,
) -> Float[Array, " outcomes"]:
    """Calculate STOP and first-winner probabilities by a unit deadline.

    Parameters
    ----------
    drifts : floating array
        Signed drift of each independent, unit-noise accumulator, starting at zero.
    eligible : Boolean array
        One entry per item. Only eligible items participate in the race.
    boundary : positive float
        Common distance to threshold in units of diffusion noise over one window.

    Returns
    -------
    floating array
        STOP followed by one probability per item. STOP uses the analytic
        survival probability. Gauss–Legendre quadrature allocates the remaining
        mass in proportion to the integrals of the first-winner densities.
    """
    drifts = jnp.asarray(drifts, dtype=jnp.result_type(drifts, boundary))

    if drifts.size == 0:
        return jnp.ones(1, dtype=drifts.dtype)

    def race():
        drift = jnp.where(eligible, drifts, 0.0)[:, None]
        log_stop = jnp.sum(
            jnp.where(eligible, _log_survival(boundary, drift[:, 0], 1.0), 0.0)
        )
        # Concentrate nodes around the fastest runner's first-passage density.
        fastest = jnp.max(jnp.where(eligible, drifts, -jnp.inf))
        mode = 2 * boundary**2 / (3 + jnp.hypot(3, 2 * boundary * fastest))
        nodes = jnp.asarray(_NODES, dtype=drifts.dtype)
        denominator = 1 - nodes + mode
        time = mode * nodes / denominator
        jacobian = mode * (1 + mode) / denominator**2
        log_weights = jnp.log(jnp.asarray(_WEIGHTS, dtype=drifts.dtype) * jacobian)
        survival = jnp.where(
            eligible[:, None], _log_survival(boundary, drift, time), 0.0
        )
        log_density = (
            jnp.log(boundary)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 1.5 * jnp.log(time)
            - 0.5 * (boundary - drift * time) ** 2 / time
        )
        log_wins = logsumexp(
            log_weights + log_density + jnp.sum(survival, axis=0) - survival, axis=1
        )
        shares = softmax(jnp.where(eligible, log_wins, -jnp.inf))
        return jnp.append(jnp.exp(log_stop), -jnp.expm1(log_stop) * shares)

    return lax.cond(
        jnp.any(eligible),
        race,
        lambda: jnp.append(jnp.ones(1, dtype=drifts.dtype), jnp.zeros_like(drifts)),
    )


class FiniteWindowRace(Pytree):
    """Use model evidence times race_drift_scale as drift toward race_boundary.

    The boundary is positive. Time and diffusion noise are measured in units
    of one retrieval window. Neither a successful recall nor its probability
    is conditioned on some accumulator reaching the boundary.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.boundary = parameters["race_boundary"]
        self.drift_scale = parameters["race_drift_scale"]

    def outcome_probabilities(
        self, model: SupportsItemEvidence
    ) -> Float[Array, " outcomes"]:
        return finite_window_probabilities(
            self.drift_scale * model.item_evidence(),
            model.is_active & model.recallable,
            self.boundary,
        )
