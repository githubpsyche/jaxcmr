import jax.numpy as jnp
import numpy as np
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
    present_lists: Float[Array, " trials study_events"],
    connections: Float[Array, " word_pool items word_pool_items"],
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices aligned to study lists.

    Args:
      present_lists: Study lists indexed by trial, containing 1-indexed item ids.
      connections: Wordpool-wide similarity matrix.
    """
    lists_np = np.array(present_lists)

    base = np.array(connections, dtype=np.float32)
    np.fill_diagonal(base, 0.0)
    base = np.maximum(base, float(lb))
    np.fill_diagonal(base, 0.0)

    trial_blocks = []
    for present in lists_np:
        valid = present > 0
        zero_based = np.where(valid, present - 1, 0)
        block = base[np.ix_(zero_based, zero_based)]
        block = np.where(
            np.logical_and(valid[:, None], valid[None, :]),
            block,
            0.0,
        )
        trial_blocks.append(jnp.array(block, dtype=jnp.float32))
    return jnp.stack(trial_blocks)
