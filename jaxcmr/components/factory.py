"""Pre-experimental semantic connection factory.

Builds per-trial semantic connection matrices from item feature
vectors for models that use pre-experimental associations.

"""

from typing import Optional

import jax.numpy as jnp
import numpy as np

from jaxcmr.math import cosine_similarity_matrix, lb
from jaxcmr.typing import (
    Array,
    Float,
)

__all__ = [
    "build_trial_connections",
    "build_trial_connections_from_similarity",
]


def build_trial_connections(
    present_lists: np.ndarray,
    features: Optional[Float[Array, " word_pool_items features_count"]],
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices aligned to study lists.

    Args:
      present_lists: Study lists indexed by trial, containing 1-indexed item ids.
      features: Wordpool-wide feature matrix or ``None`` to disable semantics.
    """

    # If no connections are provided, return zero matrices
    if features is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    connections = cosine_similarity_matrix(features)

    # Clip to non-negative values and zero the diagonal
    clipped = jnp.clip(connections, a_min=lb, a_max=None)
    zeroed = clipped * (1.0 - jnp.eye(clipped.shape[0]))

    # Extract trial-specific submatrices
    trial_blocks: list[jnp.ndarray] = []
    for trial_idx in range(present_lists.shape[0]):
        present = present_lists[trial_idx]
        valid = present > 0
        zero_based = jnp.array(jnp.where(valid, present - 1, 0), dtype=jnp.int32)
        block = zeroed[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, block, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)


def build_trial_connections_from_similarity(
    present_lists: np.ndarray,
    similarity_matrix: Optional[Float[Array, " word_pool_items word_pool_items"]],
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices from a precomputed similarity matrix.

    Args:
      present_lists: Study lists indexed by trial, containing 1-indexed item ids.
      similarity_matrix: Wordpool-wide similarity matrix or ``None`` to disable
        semantics.
    """

    if similarity_matrix is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    connections = jnp.asarray(similarity_matrix)

    # Clip to non-negative values and zero the diagonal without repairing NaNs
    # elsewhere in the matrix.
    clipped = jnp.clip(connections, a_min=lb, a_max=None)
    zeroed = jnp.where(jnp.eye(clipped.shape[0], dtype=bool), 0.0, clipped)

    trial_blocks: list[jnp.ndarray] = []
    for trial_idx in range(present_lists.shape[0]):
        present = present_lists[trial_idx]
        valid = present > 0
        zero_based = jnp.array(jnp.where(valid, present - 1, 0), dtype=jnp.int32)
        block = zeroed[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, block, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)
