"""Utilities for handling item repetitions in study lists.

Provides helpers to map recalled items to all of their valid study positions,
drop repeated recalls within trials, relabel recalls to the first occurrence
when items repeat at study, and build a shuffled control dataset for
mixed/pure-list comparisons.

Methodological stance — shuffled controls for mixed vs pure
-----------------------------------------------------------

Goal: estimate the transition rates one would expect **absent** repeated items,
under the same serial-position scaffolding used by mixed lists.

Implementation used here:
* For each subject with both mixed and pure trials, we randomly permute that
  subject's **pure-list recall rows** and pair them with the subject's **mixed-list
  presentation rows**. We repeat this pairing many times (e.g., 100 shuffles) and
  aggregate across the shuffles to form a baseline expectation.
* When a mixed-list item was studied at two positions, the control analysis
  treats recall of **either** position as recall of the same logical item. In other
  words, the null hypothesis is that the two study positions are equivalent for
  that item. Operationally, we **collapse positional codes to the first occurrence**
  (see `relabel_trial_to_firstpos`), so both study positions map to one code.
* Within-trial **repeated recalls** of the same logical item are optionally
  **zeroed** (kept only on first mention; see `filter_repeated_recalls`). This
  mirrors the usual gate in transition analyses that tabulates only while an item
  remains "not yet recalled" on the current trial.

Rationale:
* Counting **either** position for a repeater in controls avoids mechanically
  depressing the control transition rates relative to the mixed analysis. If one
  were to count only a single, pre-specified position, the control baseline will
  be lower by construction—biasing toward "detecting" an effect of repetition.
  Our approach tests the stricter null that the two positions are functionally
  the same item (encoding and retrieval treat them as such).

Key helpers:
* `relabel_trial_to_firstpos`: collapses repeated study positions to a single
  positional code per item (implements the "either-position counts" policy).
* `filter_repeated_recalls`: zeros within-trial repeated recalls, keeping first.
* `make_control_dataset`: builds the per-subject shuffled control dataset aligned
  to mixed-list presentations, applying the above normalization steps.

Terminology:
* "Mixed" lists contain repeated items; "pure" lists do not.
"""

import math
from functools import partial

import jax
import numpy as np
from jax import lax, random
from jax import numpy as jnp

from .helpers import generate_trial_mask
from .typing import Array, Int_, Integer, RecallDataset


def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Returns one-indexed positions where ``item`` appears in ``presentation``.

    Args:
      item: Item identifier; ``0`` is treated as no item.
      presentation: 1D sequence of item identifiers for the study list. Shape [list_length].
      size: Maximum number of positions to return; pads with 0 when fewer.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def all_study_positions(
    study_position: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Returns study positions of the item shown at ``study_position``.

    Args:
      study_position: One-indexed study position; ``<=0`` returns all zeros.
      presentation: 1D sequence of item identifiers for the study list. Shape [list_length].
      size: Maximum number of positions to return; pads with 0 when fewer.
    """
    item = lax.cond(
        study_position > 0,
        lambda: presentation[study_position - 1],
        lambda: 0,
    )
    return item_to_study_positions(item, presentation, size)


def filter_repeated_recalls(recalls: jnp.ndarray) -> jnp.ndarray:
    """Returns a copy with within-trial repeats set to 0, keeping first mentions.

    Use this to mirror the analysis gate that only tabulates transitions while an
    item remains "not yet recalled" on the current trial. In controls, this also
    prevents double-counting when distinct control items have been merged into the
    same logical item by positional relabeling.

    Args:
      recalls: 2D matrix of recall indices (1-indexed; 0 for no recall).
        Shape [trials, recall_positions].
    """
    n_positions = recalls.shape[1]
    matches = recalls[:, None, :] == recalls[:, :, None]
    lower_tri = jnp.tril(jnp.ones((n_positions, n_positions), bool), k=-1)
    seen_before = jnp.any(matches & lower_tri[None], axis=2)
    keep_mask = (recalls != 0) & ~seen_before
    return recalls * keep_mask


def relabel_trial_to_firstpos(
    rec_row: jnp.ndarray, pres_row: jnp.ndarray
) -> jnp.ndarray:
    """Returns recalls remapped to the **first** occurrence of their item in ``pres_row``.

    Implements the control-analysis policy that, for items repeated at study,
    **either** study position should be treated as the same logical item. Collapsing
    both positions to a single positional code avoids artificially lowering the
    control baseline relative to the mixed analysis.

    Args:
      rec_row: 1D recall indices for a single trial (1-indexed; 0 for no recall).
      pres_row: 1D study sequence of item identifiers for that trial.

    Returns:
      1D index array (same shape as ``rec_row``) with zeros preserved and all
      nonzero indices mapped to the first occurrence of that item in ``pres_row``.
    """
    L = pres_row.shape[0]
    idx = jnp.arange(L) + 1
    eq = pres_row[:, None] == pres_row[None, :]
    first_idx = jnp.min(jnp.where(eq, idx[None, :], L + 1), axis=1)
    return jnp.where(rec_row == 0, 0, first_idx[rec_row - 1])


@partial(jax.jit, static_argnums=(2,))
def _shuffle_and_tile_controls(
    control_recalls: jnp.ndarray,  # [n_pure_trials, n_recalls]
    mixed_presentations: jnp.ndarray,  # [n_mixed_trials, n_pres]
    n_permutations: int,  # static
    prng_key: jnp.ndarray,  # single PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate shuffled control recalls and align them with mixed-list presentations.

    Each permutation randomly reorders the **rows** of ``control_recalls`` (within
    subject), then we tile/truncate to match the number of ``mixed_presentations``
    rows. This produces the row-aligned control recalls used to estimate the null.

    Args:
      control_recalls: Pure-list recall matrix to permute by trial.
      mixed_presentations: Mixed-list presentation matrix.
      n_permutations: Total permutations per pure-trial block.
      prng_key: PRNG key for shuffling.

    Returns:
      (shuffled_recalls, tiled_presentations)
    """
    keys = random.split(prng_key, n_permutations)
    batched = jax.vmap(lambda k: random.permutation(k, control_recalls, axis=0))(keys)
    flat_shuffled = batched.reshape((-1, control_recalls.shape[1]))

    tiled_pres = jnp.repeat(mixed_presentations, repeats=n_permutations, axis=0)[
        : flat_shuffled.shape[0]
    ]
    flat_shuffled = flat_shuffled[: tiled_pres.shape[0]]
    # assert tiled_pres.shape[0] == flat_shuffled.shape[0]
    # assert tiled_pres.shape[1] == mixed_presentations.shape[1]
    return flat_shuffled, tiled_pres


def make_control_dataset(
    data: RecallDataset,
    mixed_query: str,
    control_query: str,
    n_shuffles: int,
    remove_repeats: bool = True,
    seed: int = 0,
) -> RecallDataset:
    """Build a shuffled-control dataset aligned to mixed-list presentations.

    Returns a per-subject control set that matches the **mixed-list serial positions**
    but uses **pure-list recall rows** permuted within subject. This encodes the null
    hypothesis that for mixed-list repeaters, **either study position counts as the
    same logical item** in controls. Operationally, control recalls are relabeled to
    the first occurrence of each mixed-list item's positional code; optional
    within-trial repeats are zeroed to mirror the main analysis gate.

    Args:
      data: Dataset containing at least ``"subject"``, ``"recalls"``, and ``"pres_itemnos"``.
      mixed_query: Boolean query string (evaluated by ``generate_trial_mask``) selecting mixed trials.
      control_query: Boolean query string selecting pure trials.
      n_shuffles: Number of shuffled control **blocks per subject** (each block permutes that
        subject’s pure-list recall rows). Total permutations per block is internally
        expanded to cover mixed rows (``ceil(n_mixed / n_pure)``).
      remove_repeats: If True, drop repeated recalls within trials in the result (keeps first mention).
      seed: RNG seed for shuffling reproducibility.

    Returns:
      RecallDataset with rows ``(#mixed_trials_per_subject * n_shuffles)`` per included subject.
      Shapes match the mixed-list presentations after tiling and truncation.
    """

    # 1) find which subjects actually have mixed trials
    all_subject_ids = jnp.array(data["subject"]).flatten()
    mixed_mask = generate_trial_mask(data, mixed_query)
    pure_mask = generate_trial_mask(data, control_query)
    subjects = np.unique(all_subject_ids[mixed_mask])
    prng_keys = random.split(random.PRNGKey(seed), subjects.size)

    recalls_blocks = []
    pres_blocks = []
    subject_id_blocks = []
    other_fields_acc = {
        key: [] for key in data if key not in ("recalls", "pres_itemnos", "subject")
    }

    for i, subj in enumerate(subjects):
        sel_pure = (all_subject_ids == subj) & pure_mask
        sel_mixed = (all_subject_ids == subj) & mixed_mask

        pure_recalls = jnp.array(data["recalls"][sel_pure])
        mixed_pres = jnp.array(data["pres_itemnos"][sel_mixed])
        if pure_recalls.shape[0] == 0 or mixed_pres.shape[0] == 0:
            continue

        n_pure, _ = pure_recalls.shape
        n_mixed, _ = mixed_pres.shape
        repeat_factor = math.ceil(n_mixed / n_pure)
        n_permutations = n_shuffles * repeat_factor

        new_recalls, new_pres = _shuffle_and_tile_controls(
            pure_recalls, mixed_pres, n_permutations, prng_keys[i]
        )
        # Ensure identical items share a single positional code
        new_recalls = jax.vmap(relabel_trial_to_firstpos, in_axes=(0, 0))(
            new_recalls, new_pres
        )
        if remove_repeats:
            new_recalls = filter_repeated_recalls(new_recalls)

        recalls_blocks.append(new_recalls)
        pres_blocks.append(new_pres)
        subject_id_blocks.append(jnp.full((new_recalls.shape[0], 1), subj, dtype=int))

        # carry along all other fields, repeated to match new_recalls
        for field, acc in other_fields_acc.items():
            arr = jnp.array(data[field])[sel_mixed]
            acc.append(
                jnp.repeat(arr, repeats=n_permutations, axis=0)[: new_recalls.shape[0]]
            )
            assert acc[-1].shape[0] == new_recalls.shape[0]
            assert acc[-1].shape[0] == new_pres.shape[0]

    return {
        "subject": jnp.vstack(subject_id_blocks),
        "recalls": jnp.vstack(recalls_blocks),
        "pres_itemnos": jnp.vstack(pres_blocks),
        **{f: jnp.vstack(lst) for f, lst in other_fields_acc.items()},
    }  # type: ignore
