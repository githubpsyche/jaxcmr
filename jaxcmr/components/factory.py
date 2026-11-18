"""A trial records items in study order. A model looks up a recalled item
by its within-list code. A word pool uses a third index, the item's
vocabulary ID.

For example, study codes `[2, 0, 1, 2]` and vocabulary IDs
`[17, 0, 42, 17]` describe three presentations and one empty slot.
With three item slots, `global_item_ids_by_code` returns `[42, 17, 0]`.
Pass that row to a connection builder to get a matrix whose first row and
column belong to local item 1. The source-category lookup uses the same
local item order for emotional context.
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
    """Gather the valence category of each item in local code order.

    Parameters
    ----------
    dataset : RecallDataset
        `pres_itemnos` and `valence` align by study position. Only positive item
        codes contribute.
    signed : bool
        Use valence signs (−1, 0, +1) for CMR3. With `False`, eCMR groups both
        nonzero signs into category 1.

    Returns
    -------
    integer array, shape (trials, item_count)
        Column `c - 1` holds the category for local code `c`. Neutral and absent
        items both have value zero. The width is the larger of the maximum list
        length and the largest code.

    Raises
    ------
    ValueError
        Presentations of the same item within a trial disagree after category coding.
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
    """Look up each local item's vocabulary ID at its first presentation.

    Parameters
    ----------
    item_codes : integer array, shape (trials, study_events)
        One-based within-list codes in study order. Zero marks an empty position.
    global_ids : integer array, shape (trials, study_events)
        Vocabulary IDs aligned with the same positions. Repeated presentations of
        one local item should carry the same ID.
    item_count : int
        Number of local item codes represented in each output row.

    Returns
    -------
    integer array, shape (trials, item_count)
        Column `c - 1` holds the first vocabulary ID for local code `c`, or zero
        if that code was absent.
    """
    codes = jnp.arange(1, item_count + 1)[None, :]
    positions = recall_items_to_positions(codes, item_codes)
    ids = jnp.take_along_axis(global_ids, jnp.maximum(positions - 1, 0), axis=1)
    return jnp.where(positions > 0, ids, 0)


def build_trial_connections(
    present_lists: Integer[ArrayLike, "trials study_events"],
    features: Float[Array, " word_pool_items features_count"] | None,
) -> Float[Array, " trials study_events study_events"]:
    """Build one trial matrix from cosine similarities between vocabulary feature rows.

    Parameters
    ----------
    present_lists : integer array, shape (trials, slots)
        One-based vocabulary IDs in the desired output order. Zero denotes padding.
    features : floating array, shape (vocabulary_size, features_count), or None
        Row `g - 1` describes vocabulary item `g`. `None` produces all-zero matrices.

    Returns
    -------
    floating array, shape (trials, slots, slots)
        Both axes follow the input row, including repeats. For finite features,
        equal IDs and padding have zero connections. Connections between distinct
        IDs have a floor of `FLOAT32_EPS`.
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
    """Gather one trial matrix from a supplied vocabulary similarity matrix.

    Parameters
    ----------
    present_lists : integer array, shape (trials, slots)
        One-based vocabulary IDs in the desired output order. Zero denotes padding.
    similarity_matrix : floating array, shape (vocabulary_size, vocabulary_size), or None
        Entry `[g - 1, h - 1]` gives the supplied similarity from item `g` to item
        `h`. `None` produces all-zero matrices.

    Returns
    -------
    floating array, shape (trials, slots, slots)
        Both axes follow the input row. Padding and same-ID pairs are zero.
        Remaining finite entries have a floor of `FLOAT32_EPS`. The orientation of
        the supplied matrix is preserved.
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
