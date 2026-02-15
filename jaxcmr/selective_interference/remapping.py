"""Recall remapping for serial position curves.

Raw recall sequences from ``sweep.batched_sweep`` use item IDs from
the full pre-allocated tier array, which may contain zero-padded gaps
where unused sweep slots sit between active phases.  Before computing
an SPC, these gaps must be closed so that item IDs map to contiguous
study positions.

``remap_recalls`` is the generic engine: given the number of active
items and allocated slots for each phase, it shifts item IDs downward
to fill gaps and optionally collapses the break phase entirely.

The four tier-specific wrappers — ``standard_remap``,
``interference_extended_remap``, ``filler_extended_remap``,
``break_extended_remap`` — plug in the correct slot counts for each
sweep tier so callers only need to pass ``(recalls, paradigm)``.

"""

from typing import Optional

import jax
import jax.numpy as jnp

from .paradigm import Paradigm


def remap_recalls(
    recalls: jax.Array,
    n_film: int,
    n_break: int,
    n_interference: int,
    *,
    n_break_slots: int,
    n_interference_slots: int,
    show_break: bool = True,
) -> jax.Array:
    """Close item-ID gaps for contiguous SPC positions.

    Parameters
    ----------
    recalls : jax.Array
        Raw recall sequences.
    n_film : int
        Number of film items.
    n_break : int
        Number of active break items.
    n_interference : int
        Number of active interference items.
    n_break_slots : int
        Allocated break-item slots in the tier.
    n_interference_slots : int
        Allocated interference-item slots in the tier.
    show_break : bool
        If False, collapse the entire break block.

    Returns
    -------
    jax.Array

    """
    break_id_lo = n_film + 1
    break_id_hi = n_film + n_break_slots
    interference_id_start = n_film + n_break_slots + 1
    filler_id_start = n_film + n_break_slots + n_interference_slots + 1

    if show_break:
        break_gap = n_break_slots - n_break
    else:
        recalls = jnp.where(
            (recalls >= break_id_lo) & (recalls <= break_id_hi), 0, recalls
        )
        break_gap = n_break_slots

    interference_gap = n_interference_slots - n_interference
    shift = jnp.where(
        recalls >= filler_id_start, break_gap + interference_gap,
        jnp.where(recalls >= interference_id_start, break_gap, 0),
    )
    return recalls - shift


def standard_remap(
    recalls: jax.Array,
    paradigm: Paradigm,
    *,
    n_break: Optional[int] = None,
    n_interference: Optional[int] = None,
    show_break: bool = True,
) -> jax.Array:
    """Remap for the standard tier.

    Parameters
    ----------
    recalls : jax.Array
        Raw recall sequences.
    paradigm : Paradigm
        Paradigm geometry.
    n_break : int, optional
        Override for active break items.
    n_interference : int, optional
        Override for active interference items.
    show_break : bool
        If False, collapse the break block.

    Returns
    -------
    jax.Array

    """
    return remap_recalls(
        recalls,
        paradigm.n_film,
        n_break if n_break is not None else paradigm.n_break,
        n_interference if n_interference is not None else paradigm.n_interference,
        n_break_slots=paradigm.n_break,
        n_interference_slots=paradigm.n_interference,
        show_break=show_break,
    )


def interference_extended_remap(
    recalls: jax.Array,
    paradigm: Paradigm,
    *,
    n_break: Optional[int] = None,
    n_interference: Optional[int] = None,
    show_break: bool = True,
) -> jax.Array:
    """Remap for the interference-extended tier.

    Parameters
    ----------
    recalls : jax.Array
        Raw recall sequences.
    paradigm : Paradigm
        Paradigm geometry.
    n_break : int, optional
        Override for active break items.
    n_interference : int, optional
        Override for active interference items.
    show_break : bool
        If False, collapse the break block.

    Returns
    -------
    jax.Array

    """
    return remap_recalls(
        recalls,
        paradigm.n_film,
        n_break if n_break is not None else paradigm.n_break,
        n_interference if n_interference is not None else paradigm.n_interference,
        n_break_slots=paradigm.n_break,
        n_interference_slots=paradigm.n_interference_max,
        show_break=show_break,
    )


def filler_extended_remap(
    recalls: jax.Array,
    paradigm: Paradigm,
    *,
    n_break: Optional[int] = None,
    n_interference: Optional[int] = None,
    show_break: bool = True,
) -> jax.Array:
    """Remap for the filler-extended tier.

    Parameters
    ----------
    recalls : jax.Array
        Raw recall sequences.
    paradigm : Paradigm
        Paradigm geometry.
    n_break : int, optional
        Override for active break items.
    n_interference : int, optional
        Override for active interference items.
    show_break : bool
        If False, collapse the break block.

    Returns
    -------
    jax.Array

    """
    return remap_recalls(
        recalls,
        paradigm.n_film,
        n_break if n_break is not None else paradigm.n_break,
        n_interference if n_interference is not None else paradigm.n_interference,
        n_break_slots=paradigm.n_break,
        n_interference_slots=paradigm.n_interference,
        show_break=show_break,
    )


def break_extended_remap(
    recalls: jax.Array,
    paradigm: Paradigm,
    *,
    n_break: Optional[int] = None,
    n_interference: Optional[int] = None,
    show_break: bool = True,
) -> jax.Array:
    """Remap for the break-extended tier.

    Parameters
    ----------
    recalls : jax.Array
        Raw recall sequences.
    paradigm : Paradigm
        Paradigm geometry.
    n_break : int, optional
        Override for active break items.
    n_interference : int, optional
        Override for active interference items.
    show_break : bool
        If False, collapse the break block.

    Returns
    -------
    jax.Array

    """
    return remap_recalls(
        recalls,
        paradigm.n_film,
        n_break if n_break is not None else paradigm.n_break,
        n_interference if n_interference is not None else paradigm.n_interference,
        n_break_slots=paradigm.n_break_max,
        n_interference_slots=paradigm.n_interference,
        show_break=show_break,
    )
