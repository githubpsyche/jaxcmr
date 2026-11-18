"""Recall data use study positions to identify recalled items.

When an item appears more than once, several positions refer to the same item.
For the study list ``[A, B, A]``, a recall recorded as position 3 refers to the
item presented at positions 1 and 3.

This module finds those positions, converts recalls to each item’s first study
position, and identifies repeated responses. Removing a repeated response
replaces its code with zero; it does not shift later responses.

``make_control_dataset`` constructs shuffled controls by pairing recall sequences
from lists without repeated items (“pure” lists) with study layouts containing
repeats (“mixed” lists). Pairings stay within participants, and each mixed trial
appears once per shuffle. The recall codes are then interpreted using the paired
mixed-list layout.
"""

from collections.abc import Collection
from typing import cast

import jax
import numpy as np
from jax import lax, random, vmap
from jax import numpy as jnp

from .dataset import generate_trial_mask, replace_recalls, select_trials
from .typing import Array, Bool, Int_, Integer, PRNGKeyArray, RecallDataset

__all__ = [
    "equivalent_study_positions",
    "filter_repeated_recalls",
    "has_repeats_per_row",
    "item_to_study_positions",
    "make_control_dataset",
    "recalls_to_first_study_positions",
]


def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Find the study positions occupied by an item.

    Parameters
    ----------
    item : Int_
        Item code to find; zero represents no item.
    presentation : Integer[Array, " list_length"]
        Item codes in study order. Equal codes identify the same item.
    size : int
        Output capacity. Keep the first ``size`` occurrences and pad unused
        entries with zero. This argument must be static under JIT.

    Returns
    -------
    Integer[Array, " size"]
        One-based study positions, in increasing order. A missing item or
        item zero produces all zeros.

    Examples
    --------
    >>> item_to_study_positions(10, jnp.array([10, 20, 10, 30]), 3).tolist()
    [1, 3, 0]
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def equivalent_study_positions(
    study_position: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Find all study positions of the item at a recalled position.

    Parameters
    ----------
    study_position : Int_
        One-based position within the study list. Nonpositive values represent
        no recall and produce zeros.
    presentation : Integer[Array, " list_length"]
        Item codes in study order. Equal codes identify the same item.
    size : int
        Output capacity, with truncation and zero padding as in
        ``item_to_study_positions``. Must be static under JIT.

    Returns
    -------
    Integer[Array, " size"]
        One-based positions of the recalled item, including the supplied
        position if it lies within the retained occurrences.

    Examples
    --------
    >>> equivalent_study_positions(3, jnp.array([10, 20, 10, 30]), 3).tolist()
    [1, 3, 0]
    """
    item = lax.cond(
        study_position > 0,
        lambda: presentation[study_position - 1],
        lambda: 0,
    )
    return item_to_study_positions(item, presentation, size)


def has_repeats_per_row(
    arr: Integer[Array, " trials recall_events"],
) -> Bool[Array, " trials"]:
    """Identify rows containing repeated positive codes.

    Parameters
    ----------
    arr : Integer[Array, " trials recall_events"]
        Codes to examine independently in each row. These may represent
        study items or recalled positions. Zero and negative codes are ignored.

    Returns
    -------
    Bool[Array, " trials"]
        True where a positive code occurs more than once. Different codes
        referring to the same item are not combined by this operation.
    """

    def check_row(row: Integer[Array, " recall_events"]) -> Bool[Array, ""]:
        valid_positions = row > 0
        padded_values = lax.select(valid_positions, row, jnp.full_like(row, -1))
        sorted_values = jnp.sort(padded_values)
        adjacent_matches = sorted_values[1:] == sorted_values[:-1]
        valid_repeats = adjacent_matches & (sorted_values[1:] != -1)
        return jnp.any(valid_repeats)

    return vmap(check_row)(arr)


def recalls_to_first_study_positions(
    rec_row: Integer[Array, " recall_events"],
    pres_row: Integer[Array, " study_events"],
) -> Integer[Array, " recall_events"]:
    """Represent each recalled item by its first study position.

    Parameters
    ----------
    rec_row : Integer[Array, " recall_events"]
        One-based recalled positions within this trial's study list, with zero
        for no recall. Positive codes must refer to presented items.
    pres_row : Integer[Array, " study_events"]
        Study item codes; equal codes identify repeated presentations.

    Returns
    -------
    Integer[Array, " recall_events"]
        Recalls mapped to each item's first study position, preserving zeros
        and response slots. Repeated recalls remain; use
        ``filter_repeated_recalls`` separately to zero later occurrences.

    Examples
    --------
    >>> recalls_to_first_study_positions(
    ...     jnp.array([3, 2, 1, 0]), jnp.array([10, 20, 10, 30])
    ... ).tolist()
    [1, 2, 1, 0]
    """
    study_length = pres_row.shape[0]
    study_positions = jnp.arange(study_length) + 1
    same_item = pres_row[:, None] == pres_row[None, :]
    first_positions = jnp.min(
        jnp.where(same_item, study_positions[None, :], study_length + 1), axis=1
    )
    return jnp.where(rec_row == 0, 0, first_positions[rec_row - 1])


def filter_repeated_recalls(
    recalls: Integer[Array, " trials recall_events"],
) -> Integer[Array, " trials recall_events"]:
    """Zero repeated recall codes while preserving response slots.

    Parameters
    ----------
    recalls : Integer[Array, " trials recall_events"]
        Recall codes, with zero for no recall. Equality of codes determines
        repetition. If different study positions refer to the same item,
        relabel with ``recalls_to_first_study_positions`` first.

    Returns
    -------
    Integer[Array, " trials recall_events"]
        A new array retaining the first occurrence of each nonzero code and
        replacing later occurrences with zero. Rows are not compacted; how
        zero slots affect a measure is determined by that analysis.

    Examples
    --------
    >>> filter_repeated_recalls(jnp.array([[1, 2, 1, 3]])).tolist()
    [[1, 2, 0, 3]]
    """
    recall_count = recalls.shape[1]
    matches = recalls[:, None, :] == recalls[:, :, None]
    earlier_positions = jnp.tril(jnp.ones((recall_count, recall_count), bool), k=-1)
    seen_before = jnp.any(matches & earlier_positions[None], axis=2)
    keep_mask = (recalls != 0) & ~seen_before
    return recalls * keep_mask


def _make_subject_control(
    mixed_data: RecallDataset,
    pure_recalls: Integer[Array, " pure_trials recall_events"],
    shuffle_count: int,
    subject_key: PRNGKeyArray,
    remove_repeats: bool,
    recall_dependent_fields: Collection[str],
) -> RecallDataset:
    """Pair pure recalls with every mixed trial once per shuffle for one participant."""
    mixed_count = mixed_data["pres_itemnos"].shape[0]
    pure_count = pure_recalls.shape[0]
    permutations_per_shuffle = (mixed_count + pure_count - 1) // pure_count
    keys = random.split(subject_key, shuffle_count * permutations_per_shuffle)
    pure_indices = jax.vmap(lambda key: random.permutation(key, pure_count))(keys)
    # Each shuffle starts fresh; only its final permutation may be truncated.
    pure_indices = pure_indices.reshape(shuffle_count, -1)[:, :mixed_count].reshape(-1)
    mixed_indices = jnp.tile(jnp.arange(mixed_count), shuffle_count)
    control = select_trials(mixed_data, mixed_indices)
    control_recalls = jax.vmap(recalls_to_first_study_positions)(
        pure_recalls[pure_indices], control["pres_itemnos"]
    )
    if remove_repeats:
        control_recalls = filter_repeated_recalls(control_recalls)
    if "listLength" not in control:
        control["listLength"] = jnp.count_nonzero(
            control["pres_itemnos"], axis=1, keepdims=True
        )
    return replace_recalls(control, control_recalls, recall_dependent_fields)


def make_control_dataset(
    data: RecallDataset,
    mixed_query: str,
    control_query: str,
    n_shuffles: int,
    remove_repeats: bool = True,
    seed: int = 0,
    recall_dependent_fields: Collection[str] = (),
) -> RecallDataset:
    """Pair shuffled pure-list recalls with mixed-list presentations by participant.

    Parameters
    ----------
    data : RecallDataset
        Trial data containing both conditions. Recall codes are one-based study
        positions; pure and mixed trials must use compatible position codes.
        Study metadata is taken from mixed trials, not from the pure trials
        supplying the recalls. The input is unchanged.
    mixed_query : str
        Query selecting lists with repeated study items, evaluated by
        ``generate_trial_mask``.
    control_query : str
        Query selecting pure lists without repeated study items.
    n_shuffles : int
        Positive number of shuffles per participant. Each shuffle pairs every
        mixed trial once with a pure recall row. Pure rows are drawn in random
        permutation order; use additional independent permutations when there
        are fewer pure trials, truncating only the last permutation.
    remove_repeats : bool, optional
        Whether to zero repeated recalls after relabeling each item to its
        first mixed-list study position. Default is True.
    seed : int, optional
        Random seed for reproducible pairings. Default is zero.
    recall_dependent_fields : Collection[str], optional
        Additional response fields to discard when replacing recalls. Known
        response fields, including ``rec_itemids``, ``irt``, ``cue_clips`` and
        ``_should_tabulate``, are also removed.

    Returns
    -------
    RecallDataset
        Controls grouped by participant, then shuffle, then mixed-trial order.
        Each included participant contributes ``n_shuffles * mixed_trials``
        rows. Participants without both conditions are skipped. Study fields
        remain aligned; list lengths are inferred from nonzero presentations
        when absent from legacy input data.

    Raises
    ------
    ValueError
        If no participant has both mixed and pure trials.

    Notes
    -----
    The control preserves pure-list recall sequences but interprets their
    position codes under mixed-list presentations. Positions occupied by the
    same mixed-list item become a single code. This is an explicit control
    construction; it does not estimate a repetition effect or perform a test.
    Shuffles are repeated pairings within participants, not new participants.
    """
    all_subject_ids = data["subject"].flatten()
    mixed_mask = generate_trial_mask(data, mixed_query)
    pure_mask = generate_trial_mask(data, control_query)
    subjects = np.unique(all_subject_ids[mixed_mask])
    subject_keys = random.split(random.PRNGKey(seed), subjects.size)
    subject_controls = []

    for subject_index, subject_id in enumerate(subjects):
        pure_selection = (all_subject_ids == subject_id) & pure_mask
        mixed_selection = (all_subject_ids == subject_id) & mixed_mask
        pure_recalls = data["recalls"][pure_selection]
        if pure_recalls.shape[0] == 0:
            continue
        mixed_data = select_trials(data, jnp.flatnonzero(mixed_selection))
        subject_controls.append(_make_subject_control(
            mixed_data, pure_recalls, n_shuffles, subject_keys[subject_index],
            remove_repeats, recall_dependent_fields,
        ))

    if not subject_controls:
        raise ValueError("No participants have both mixed and pure trials.")

    # Concatenate trial rows, including one-dimensional custom fields.
    return cast(RecallDataset, {
        field: jnp.concatenate([control[field] for control in subject_controls], axis=0)
        for field in subject_controls[0]
    })
