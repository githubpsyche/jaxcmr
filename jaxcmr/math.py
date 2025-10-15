from typing import Optional

import jax.numpy as jnp
from jax import lax

from jaxcmr.typing import Array, Float, Float_, Int_

lb = jnp.finfo(jnp.float32).eps


def power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value scaled by the exponent factor using logsumexp trick."""
    log_activation = jnp.log(value)
    return lax.cond(
        jnp.logical_and(jnp.any(value != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: value,
        None,
    )


def simple_power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value raised to the specified power without logsumexp trick."""
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


def exponential_stop_probability(
    stop_probability_scale: Float_, stop_probability_growth: Float_, recall_total: Int_
):
    """Returns the exponential stop probability for the specified recall event.

    Args:
        stop_probability_scale: the scale factor for stop probability.
        stop_probability_growth: the growth factor for stop probability.
        recall_total: the total number of items recalled.
    """
    return stop_probability_scale * jnp.exp(recall_total * stop_probability_growth)


def normalize_magnitude(
    vector: Float[Array, " features"],
) -> Float[Array, " features"]:
    """Return the input vector normalized to unit length."""
    return vector / jnp.sqrt(jnp.sum(vector**2) + lb)


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


def build_trial_connections(
    present_lists: Array,
    connections: Optional[Float[Array, " word_pool_items word_pool_items"]],
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices aligned to study lists.

    Args:
      present_lists: Study lists indexed by trial, containing 1-indexed item ids.
      connections: Wordpool-wide similarity matrix or ``None`` to disable semantics.
    """

    # If no connections are provided, return zero matrices
    if connections is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    # Clip to non-negative values and zero the diagonal
    clipped = jnp.clip(connections, a_min=lb, a_max=None)
    zeroed = clipped * (1.0 - jnp.eye(clipped.shape[0]))

    # Extract trial-specific submatrices
    trial_blocks: list[jnp.ndarray] = []
    for trial_idx in range(present_lists.shape[0]):
        present = present_lists[trial_idx]
        valid = present > 0
        zero_based = jnp.where(valid, present - 1, 0).astype(jnp.int32)
        block = zeroed[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, block, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)
