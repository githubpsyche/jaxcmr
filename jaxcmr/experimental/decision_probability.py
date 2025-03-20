import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.scipy.special import ndtr
from jaxtyping import Array, Bool, Float, Integer
from simple_pytree import Pytree

from jaxcmr.math import lb

Float_ = Float[Array, ""] | float | int
Int_ = Integer[Array, ""] | int
Bool_ = Bool[Array, ""] | bool

theta = 200.0
tmax = 400.0
n_steps = 400
t_vals = np.maximum(np.linspace(0.0, tmax, n_steps), lb)
dt = t_vals[1] - t_vals[0]


def wald_pdf(v: Float_) -> Float[Array, " timepoints"]:
    """
    Returns the Wald (Inverse Gaussian) PDF at each time point.

    Wald PDF:
      f(t) = theta / sqrt(2π * t^3) * exp( - (theta - v * t)^2 / (2 * t) )
    """
    numerator = theta
    denominator = jnp.sqrt(2.0 * jnp.pi * t_vals**3)
    exponent = -((theta - v * t_vals) ** 2) / (2.0 * t_vals)
    exponent = jnp.clip(exponent, a_min=None, a_max=50.0)
    return (numerator / denominator) * jnp.exp(exponent)


def wald_cdf_closed_form(v: Float_) -> Float[Array, " timepoints"]:
    t = jnp.maximum(t_vals, 1e-30)
    alpha = (v * t - theta) / jnp.sqrt(t)
    beta  = -(v * t + theta) / jnp.sqrt(t)
    exp_term = 2.0 * theta * v
    # Clip to avoid overflow/underflow
    exp_term = jnp.clip(exp_term, -60.0, 60.0)
    return ndtr(alpha) + jnp.exp(exp_term) * ndtr(beta)


def race_diffusion_precompute(
    drifts: Float[Array, " runners"],
) -> tuple[
    Float_, Float[Array, " runners timepoints"], Float[Array, " runners timepoints"]
]:
    """
    Precompute the time grid, PDFs, and CDFs for each runner.

    Returns:
        dt: The spacing between time points.
        pdfs: PDFs for each runner (shape (n_runners, n_steps)).
        cdfs: CDFs for each runner (shape (n_runners, n_steps)).
    """

    def single_runner_pdf_cdf(v):
        pdf_v = wald_pdf(v)
        # cdf_v = wald_cdf_from_pdf(pdf_v)
        cdf_v = wald_cdf_closed_form(v)
        return pdf_v, cdf_v

    pdfs, cdfs = vmap(single_runner_pdf_cdf, in_axes=(0,))(
        drifts
    )  # shape: (n_runners, n_steps)
    return dt, pdfs, cdfs


def compute_runner_probability(
    i: Int_,
    pdfs: Float[Array, "runners timepoints"],
    cdfs: Float[Array, " runners timepoints"],
    dt: Float_,
) -> Float_:
    """
    Probability that runner i finishes first, i.e.:
      ∫ [pdfs[i,t] * ∏_{j != i} (1 - cdfs[j,t])] dt
    Using a single product over all runners' (1 - cdfs[j,t]) and dividing out the i-th factor.
    """
    # ∏_{j} [1 - cdfs[j,t]]
    all_prod = jnp.prod(1.0 - cdfs, axis=0)
    # Avoid division by zero if (1 - cdfs[i,t]) == 0
    denom = jnp.clip(1.0 - cdfs[i], 1e-30, None)
    density_i = pdfs[i] * (all_prod / denom)
    return jnp.trapezoid(density_i, dx=dt)


def compute_runner_probabilities(
    pdfs: Float[Array, "runners timepoints"],
    cdfs: Float[Array, " runners timepoints"],
    dt: Float_,
) -> Float[Array, " runners"]:
    """
    Returns finishing-first probabilities for all runners, normalized.

    p_i = ∫ pdfs[i,t] * ( ∏_{j} [1 - cdfs[j,t]] ) / (1 - cdfs[i,t]) dt
    """
    all_prod = jnp.prod(1.0 - cdfs, axis=0)

    def _prob_for_runner(i):
        denom = jnp.clip(1.0 - cdfs[i], 1e-30, None)
        density_i = pdfs[i] * (all_prod / denom)
        return jnp.trapezoid(density_i, dx=dt)

    raw_probs = vmap(_prob_for_runner)(jnp.arange(pdfs.shape[0]))
    total = jnp.sum(raw_probs)
    return lax.cond(total > 0, lambda: raw_probs / total, lambda: raw_probs)


class RaceDiffusionModel(Pytree):
    def outcome_probability(
        self,
        item_index: Int_,
        supports: Float[Array, " items"],
    ) -> Float_:
        """
        Probability of selecting the specified item (index).
        """
        d, pdfs, cdfs = race_diffusion_precompute(supports)
        return compute_runner_probability(item_index, pdfs, cdfs, d)

    def outcome_probabilities(
        self,
        supports: Float[Array, " items"],
    ) -> Float[Array, " items"]:
        """
        Returns probability distribution over all items.
        """
        d, pdfs, cdfs = race_diffusion_precompute(supports)
        return compute_runner_probabilities(pdfs, cdfs, d)
