"""Trial-specific source categories and pre-experimental semantic connections.

Model factories align study metadata with local item codes before constructing
source associations and semantic connection matrices.

"""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np

from jaxcmr.dataset import recall_items_to_positions
from jaxcmr.math import FLOAT32_EPS, cosine_similarity_matrix
from jaxcmr.typing import Array, ArrayLike, Float, Integer, RecallDataset

__all__ = [
    "build_trial_connections",
    "build_trial_connections_from_similarity",
    "global_item_ids_by_code",
    "source_categories_by_item",
]


def source_categories_by_item(
    dataset: RecallDataset, *, signed: bool
) -> Integer[Array, "trials item_count"]:
    """Return the source category of each local item, in model code order.

    eCMR groups positive and negative items together (1). CMR3 keeps their
    signs. A presented item may be neutral (0). Empty study slots contribute
    nothing. All presentations of an item must agree on its category because
    retrieval assigns that item a single source association. The row width
    accommodates both the largest local code and all study events (including
    repeated presentations) used to index primacy.
    """
    codes = np.asarray(dataset["pres_itemnos"])
    trials, slots = np.nonzero(codes > 0)
    items = codes[trials, slots] - 1
    valence = np.asarray(cast(Mapping[str, Array], dataset)["valence"])[trials, slots]
    categories = np.sign(valence) if signed else valence != 0
    item_count = max(int(np.max(dataset["listLength"])), int(codes.max(initial=0)))
    by_item = np.zeros((codes.shape[0], item_count), dtype=np.int8)
    by_item[trials, items] = categories
    if np.any(by_item[trials, items] != categories):
        raise ValueError("Repeated presentations of an item have conflicting source categories.")
    return jnp.asarray(by_item)


def global_item_ids_by_code(
    item_codes: Integer[Array, "trials study_events"],
    global_ids: Integer[Array, "trials study_events"],
    item_count: int,
) -> Integer[Array, "trials item_count"]:
    """Look up each model item's vocabulary ID in its trial's study list.

    Column `c - 1` holds the global ID at the first presentation of local
    code `c`, or zero if the item was never presented. Repeated presentations
    of one item share a vocabulary ID. Pass these rows to a trial connection
    builder when the model indexes semantic support by local item code.
    """
    codes = jnp.arange(1, item_count + 1)[None, :]
    positions = recall_items_to_positions(codes, item_codes)
    ids = jnp.take_along_axis(global_ids, jnp.maximum(positions - 1, 0), axis=1)
    return jnp.where(positions > 0, ids, 0)


def build_trial_connections(
    present_lists: Integer[ArrayLike, "trials study_events"],
    features: Float[Array, " word_pool_items features_count"] | None,
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices aligned to study lists.

    Args:
      present_lists: Integer study lists with 1-indexed global item IDs; zero
          denotes padding. Positive IDs must be covered by the supplied vocabulary.
      features: Wordpool-wide feature matrix or ``None`` to disable semantics.
    """

    present_lists = jnp.asarray(present_lists)

    # If no connections are provided, return zero matrices
    if features is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    connections = cosine_similarity_matrix(features)

    # Clip to non-negative values and zero the diagonal
    nonnegative_connections = jnp.clip(connections, a_min=FLOAT32_EPS, a_max=None)
    connections_without_diagonal = nonnegative_connections * (1.0 - jnp.eye(nonnegative_connections.shape[0]))

    # Extract trial-specific submatrices
    trial_blocks: list[jnp.ndarray] = []
    for trial_index in range(present_lists.shape[0]):
        present = present_lists[trial_index]
        valid = present > 0
        zero_based = jnp.array(jnp.where(valid, present - 1, 0), dtype=jnp.int32)
        trial_connections = connections_without_diagonal[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, trial_connections, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)


def build_trial_connections_from_similarity(
    present_lists: Integer[ArrayLike, "trials study_events"],
    similarity_matrix: Float[Array, " word_pool_items word_pool_items"] | None,
) -> Float[Array, " trials study_events study_events"]:
    """Returns per-trial connection matrices from a precomputed similarity matrix.

    Args:
      present_lists: Integer study lists with 1-indexed global item IDs; zero
          denotes padding. Positive IDs must be covered by the supplied vocabulary.
      similarity_matrix: Square wordpool-wide similarity matrix or ``None``
          to disable semantics. Item IDs must be within its row/column coverage.
    """

    present_lists = jnp.asarray(present_lists)

    if similarity_matrix is None:
        list_length = present_lists.shape[1]
        blank = jnp.zeros((list_length, list_length))
        return jnp.stack([blank] * present_lists.shape[0])

    connections = jnp.asarray(similarity_matrix)

    # Clip to non-negative values and zero the diagonal without repairing NaNs
    # elsewhere in the matrix.
    nonnegative_connections = jnp.clip(connections, a_min=FLOAT32_EPS, a_max=None)
    connections_without_diagonal = jnp.where(jnp.eye(nonnegative_connections.shape[0], dtype=bool), 0.0, nonnegative_connections)

    trial_blocks: list[jnp.ndarray] = []
    for trial_index in range(present_lists.shape[0]):
        present = present_lists[trial_index]
        valid = present > 0
        zero_based = jnp.array(jnp.where(valid, present - 1, 0), dtype=jnp.int32)
        trial_connections = connections_without_diagonal[zero_based[:, None], zero_based[None, :]]
        keep_mask = jnp.logical_and(valid[:, None], valid[None, :])
        trial_blocks.append(jnp.where(keep_mask, trial_connections, 0.0).astype(jnp.float32))

    return jnp.stack(trial_blocks)
