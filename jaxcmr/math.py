"""Numerical primitives for memory-search models.

Provides power-scaling, normalization, primacy decay, and cosine
similarity functions used by model components and analyses.

"""

import jax.numpy as jnp
from jax import lax

from jaxcmr.typing import Array, Float, Float_, Int_

__all__ = [
    "cosine_similarity_matrix",
    "exponential_primacy_decay",
    "lb",
    "normalize_magnitude",
    "power_scale",
    "power_scale_absolute",
    "relative_power_scale",
    "simple_power_scale",
]

lb = jnp.finfo(jnp.float32).eps


def power_scale(value: Float_, scale: Float_) -> Float:
    """Legacy power transform, preserving the identity shortcut at scale one.

    Use ``relative_power_scale`` when differentiating with respect to scale:
    this historical shortcut changes normalization at exactly one.
    """
    return lax.cond(
        scale == 1,
        lambda: value,
        lambda: relative_power_scale(value, scale),
    )


def relative_power_scale(
    value: Float[Array, "..."] | float, scale: Float_
) -> Float[Array, "..."]:
    """Raise nonnegative support to a positive power relative to its maximum.

    Positive entries equal ``(value / max(value)) ** scale`` at every scale,
    including one. Zero support stays zero; all-zero input returns zeros.
    Guard log/exp inputs before masking so structural zeros have finite
    derivatives. Negative support remains invalid rather than being clipped.
    """
    nonzero = value != 0
    logs = jnp.log(jnp.where(nonzero, value, 1.0))
    maximum = jnp.max(jnp.where(nonzero, logs, -jnp.inf))
    maximum = jnp.where(jnp.any(nonzero), maximum, 0.0)
    shift = jnp.where(nonzero, logs - maximum, 0.0)
    return jnp.where(nonzero, jnp.exp(scale * shift), 0.0)


def power_scale_absolute(x, exponent, eps=lb):
    """Returns a magnitude-preserving, log-stable power transform."""
    logx = jnp.log(x + eps)
    # subtract max to stay stable
    logx_centered = logx - jnp.max(logx)
    # re-add the max so magnitude is preserved
    scaled = exponent * logx_centered + jnp.max(logx)
    return jnp.exp(scaled)


def simple_power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value raised to the specified power without stabilization."""
    return value**scale


def exponential_primacy_decay(
    study_index: Int_, primacy_scale: Float_, primacy_decay: Float_
):
    """Returns the exponential primacy weighting for the specified study event.

    Args:
        study_index: the index of the study event.
        primacy_scale: the scale factor for primacy effect.
        primacy_decay: the decay factor for primacy effect.
    """
    return primacy_scale * jnp.exp(-primacy_decay * study_index) + 1


def normalize_magnitude(
    vector: Float[Array, " features"],
) -> Float[Array, " features"]:
    """Normalize nonzero vectors; the exact-zero branch has value and tangent zero.

    Guard the square root itself, so reverse differentiation of an inactive
    zero-input branch cannot introduce NaNs into a batched model calculation.
    Normalization is mathematically undefined at zero; the zero tangent is an
    explicit convention for that branch, not a limiting derivative.
    """
    squared = jnp.sum(vector**2)
    is_zero = squared == 0
    safe_magnitude = jnp.sqrt(jnp.where(is_zero, 1.0, squared))
    return jnp.where(is_zero, jnp.zeros_like(vector), vector / safe_magnitude)


def cosine_similarity_matrix(
    features: Float[Array, " items features"],
) -> Float[Array, " items items"]:
    """Returns pairwise cosine similarities for ``features``.

    Args:
      features: Matrix whose rows are item feature vectors.
    """

    norms = jnp.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norms + lb)
    return jnp.clip(normalized @ normalized.T, -1.0, 1.0)
