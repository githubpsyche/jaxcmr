"""Numerical operations for memory-search models and likelihood calculations.

Use ``relative_power_scale`` to sharpen or flatten nonnegative support while
keeping its largest positive value at one. The caller determines what the
support represents and how the resulting weights enter the model.

``normalize_magnitude`` converts nonzero vectors to unit length.
``cosine_similarity_matrix`` applies that normalization to each row of an
item-feature matrix, then returns pairwise dot products.

``exponential_primacy_decay`` assigns weights to zero-based study positions,
with an exponential primacy effect above a baseline of one.
``negative_log_likelihood`` sums negative log probabilities over supplied
events; the caller chooses which events contribute.

These operations take numerical values or arrays directly, so they can be
used within model calculations or independently of a model.

``FLOAT32_EPS`` is float32 machine epsilon, exactly ``2**-23``. Its value stays
fixed when used in float64 calculations.
"""

import jax.numpy as jnp
from jax import vmap

from jaxcmr.typing import Array, Float, Float_, Integer

__all__ = [
    "FLOAT32_EPS",
    "cosine_similarity_matrix",
    "exponential_primacy_decay",
    "negative_log_likelihood",
    "normalize_magnitude",
    "relative_power_scale",
]

FLOAT32_EPS = jnp.finfo(jnp.float32).eps


def relative_power_scale(
    value: Float[Array, "..."] | float, scale: Float_
) -> Float[Array, "..."]:
    """Raise support to a power after dividing by its largest value.

    Parameters
    ----------
    value : float or floating array
        Finite, nonnegative support. The maximum is taken over the entire
        array; use ``jax.vmap`` to transform several vectors independently.
    scale : float or scalar floating array
        Positive exponent. Values above one sharpen the relative weights;
        values below one flatten them.

    Returns
    -------
    scaled : floating array
        Same shape as ``value``. Positive entries equal
        ``(value / max(value)) ** scale``, including at scale one. Zero
        entries remain zero, and all-zero input returns zeros.

    Notes
    -----
    The calculation uses differences of logarithms rather than computing
    large powers directly. Guarding inactive log/exp inputs keeps derivatives
    finite at structural zeros. Negative support is not clipped or repaired.

    Examples
    --------
    >>> relative_power_scale(jnp.array([2.0, 1.0, 0.0]), 2.0)
    Array([1.  , 0.25, 0.  ], dtype=float32)
    """
    nonzero = value != 0
    logs = jnp.log(jnp.where(nonzero, value, 1.0))
    maximum = jnp.max(jnp.where(nonzero, logs, -jnp.inf))
    maximum = jnp.where(jnp.any(nonzero), maximum, 0.0)
    shift = jnp.where(nonzero, logs - maximum, 0.0)
    return jnp.where(nonzero, jnp.exp(scale * shift), 0.0)


def normalize_magnitude(
    vector: Float[Array, " features"],
) -> Float[Array, " features"]:
    """Divide a vector by its Euclidean norm, leaving zero vectors at zero.

    Parameters
    ----------
    vector : floating array, shape (features,)
        Vector to normalize.

    Returns
    -------
    normalized : floating array, shape (features,)
        Unit vector with the input's direction, or zeros when its computed
        squared norm is zero.

    Notes
    -----
    The zero branch has both value and derivative zero. Normalization has no
    mathematical derivative at zero; this is an explicit convention.
    Guarding the square root prevents an inactive zero branch from introducing
    NaNs during batched reverse differentiation. Nonfinite inputs are not
    repaired.
    """
    squared = jnp.sum(vector**2)
    is_zero = squared == 0
    safe_magnitude = jnp.sqrt(jnp.where(is_zero, 1.0, squared))
    return jnp.where(is_zero, jnp.zeros_like(vector), vector / safe_magnitude)


def cosine_similarity_matrix(
    features: Float[Array, " items features"],
) -> Float[Array, " items items"]:
    """Calculate cosine similarity between every pair of item feature rows.

    Parameters
    ----------
    features : floating array, shape (items, features)
        One feature vector per item.

    Returns
    -------
    similarities : floating array, shape (items, items)
        Dot products of unit-normalized rows, clipped to ``[-1, 1]`` to
        remove floating-point overshoot. Identical directions have similarity
        one, orthogonal directions zero, and opposite directions minus one.
        A zero row has similarity zero to every row, including itself.

    Notes
    -----
    Each row is normalized by ``normalize_magnitude``, without an added
    epsilon. Rescaling a nonzero row by a positive factor preserves its
    similarities within floating-point range and precision.
    """
    normalized = vmap(normalize_magnitude)(features)
    return jnp.clip(normalized @ normalized.T, -1.0, 1.0)


def exponential_primacy_decay(
    study_index: Integer[Array, "..."] | int, primacy_scale: Float_, primacy_decay: Float_
) -> Float[Array, "..."]:
    """Calculate exponential primacy weights above a baseline of one.

    Parameters
    ----------
    study_index : int or integer array
        Zero-based study positions: the first presentation has index zero.
    primacy_scale : float or scalar floating array
        Additional weight at the first presentation.
    primacy_decay : float or scalar floating array
        Decay rate per study position. A positive rate reduces the additional
        weight as study proceeds; zero keeps it constant.

    Returns
    -------
    weights : floating array
        ``1 + primacy_scale * exp(-primacy_decay * study_index)``, with the
        shape of ``study_index``. A zero scale gives weight one everywhere.
    """
    return primacy_scale * jnp.exp(-primacy_decay * study_index) + 1


def negative_log_likelihood(
    likelihoods: Float[Array, "trial_count ..."],
) -> Float[Array, ""]:
    """Sum negative natural-log probabilities over all supplied entries.

    Parameters
    ----------
    likelihoods : floating array
        Probabilities assigned to observed events. Every dimension is summed;
        callers can exclude an event by replacing its probability with one.

    Returns
    -------
    loss : scalar floating array
        ``-sum(log(likelihoods))``. A zero probability gives infinite loss.
        An empty array gives zero loss.

    Notes
    -----
    Probabilities are neither clipped nor offset by an epsilon. Event
    selection and calculation of the probabilities belong to the caller.
    """
    return -jnp.sum(jnp.log(likelihoods))
