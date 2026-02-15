"""Selective-interference paradigm structure.

A selective interference experiment presents items in a fixed sequence
of phases — Film, Break, Interference, Filler — then tests recall.
The ``Paradigm`` dataclass captures the size of each phase and the
pre-allocated ceilings used when sweeping phase counts under JAX JIT.

Item IDs are 1-indexed and laid out contiguously::

    [1 … n_film] [n_film+1 … n_film+n_break] [… n_interference] [… n_filler]

Because JAX JIT requires fixed array shapes, parameter sweeps that
vary a phase count (e.g. 4, 8, 16 interference items) must pre-allocate
arrays at a ceiling size and zero-pad unused slots.  The ``make_*``
functions build these zero-padded arrays for each sweep tier.

``compute_n_presented`` returns the number of valid study positions
after remapping, accounting for which phases are visible in an SPC.

"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Paradigm:
    """Selective-interference paradigm geometry."""

    # Phase sizes (standard condition)
    n_film: int = 16        # target items (film clips)
    n_break: int = 16       # break items between film and reminder
    n_interference: int = 16      # interference items after reminder
    n_filler: int = 16      # filler items before recall

    # Sweep ceilings (pre-allocated slots for parameter sweeps)
    n_break_max: int = 32   # max break items across sweep conditions
    n_interference_max: int = 32  # max interference items across sweep conditions
    n_filler_max: int = 32  # max filler items across sweep conditions

    # Simulation dimensions
    max_recall: int = 48          # maximum recall attempts per trial
    experiment_count: int = 100   # replications per subject

    # -- Tier list lengths ------------------------------------------------

    @property
    def list_length(self) -> int:
        """Standard tier list length."""
        return self.n_film + self.n_break + self.n_interference + self.n_filler

    @property
    def interference_extended_list_length(self) -> int:
        """Interference-extended tier list length."""
        return self.n_film + self.n_break + self.n_interference_max + self.n_filler

    @property
    def break_extended_list_length(self) -> int:
        """Break-extended tier list length."""
        return self.n_film + self.n_break_max + self.n_interference + self.n_filler

    @property
    def filler_extended_list_length(self) -> int:
        """Filler-extended tier list length."""
        return self.n_film + self.n_break + self.n_interference + self.n_filler_max

    # -- Standard-tier item arrays ----------------------------------------

    @property
    def film_items(self) -> jax.Array:
        """Item IDs for the film phase."""
        return jnp.arange(1, self.n_film + 1)

    @property
    def break_items(self) -> jax.Array:
        """Item IDs for the break phase."""
        return jnp.arange(self.n_film + 1, self.n_film + self.n_break + 1)

    @property
    def interference_items(self) -> jax.Array:
        """Item IDs for the interference phase."""
        s = self.n_film + self.n_break + 1
        return jnp.arange(s, s + self.n_interference)

    @property
    def filler_items(self) -> jax.Array:
        """Item IDs for the filler phase."""
        s = self.n_film + self.n_break + self.n_interference + 1
        return jnp.arange(s, s + self.n_filler)

    # -- Extended-tier item arrays ----------------------------------------

    @property
    def interference_extended_filler_items(self) -> jax.Array:
        """Filler items in the interference-extended tier."""
        s = self.n_film + self.n_break + self.n_interference_max + 1
        return jnp.arange(s, s + self.n_filler)

    @property
    def break_extended_interference_items(self) -> jax.Array:
        """Interference items in the break-extended tier."""
        s = self.n_film + self.n_break_max + 1
        return jnp.arange(s, s + self.n_interference)

    @property
    def break_extended_filler_items(self) -> jax.Array:
        """Filler items in the break-extended tier."""
        s = self.n_film + self.n_break_max + self.n_interference + 1
        return jnp.arange(s, s + self.n_filler)


def make_extended_interference(paradigm: Paradigm, m: int) -> jax.Array:
    """Build zero-padded interference items for extended tier.

    Parameters
    ----------
    paradigm : Paradigm
        Paradigm geometry.
    m : int
        Number of active interference items.

    Returns
    -------
    jax.Array

    """
    s = paradigm.n_film + paradigm.n_break + 1
    all_items = jnp.arange(s, s + paradigm.n_interference_max)
    return jnp.where(jnp.arange(paradigm.n_interference_max) < m, all_items, 0)


def make_extended_break(paradigm: Paradigm, n: int) -> jax.Array:
    """Build zero-padded break items for break-extended tier.

    Parameters
    ----------
    paradigm : Paradigm
        Paradigm geometry.
    n : int
        Number of active break items.

    Returns
    -------
    jax.Array

    """
    all_items = jnp.arange(
        paradigm.n_film + 1, paradigm.n_film + paradigm.n_break_max + 1
    )
    return jnp.where(jnp.arange(paradigm.n_break_max) < n, all_items, 0)


def make_extended_filler(paradigm: Paradigm, n: int) -> jax.Array:
    """Build zero-padded filler items for filler-extended tier.

    Parameters
    ----------
    paradigm : Paradigm
        Paradigm geometry.
    n : int
        Number of active filler items.

    Returns
    -------
    jax.Array

    """
    s = paradigm.n_film + paradigm.n_break + paradigm.n_interference + 1
    all_items = jnp.arange(s, s + paradigm.n_filler_max)
    return jnp.where(jnp.arange(paradigm.n_filler_max) < n, all_items, 0)


def compute_n_presented(
    paradigm: Paradigm,
    *,
    n_break: Optional[int] = None,
    n_interference: Optional[int] = None,
    n_filler: Optional[int] = None,
    show_break: bool = True,
    show_fillers: bool = True,
) -> int:
    """Compute number of presented positions after remapping.

    Parameters
    ----------
    paradigm : Paradigm
        Paradigm geometry.
    n_break : int, optional
        Override for active break items.
    n_interference : int, optional
        Override for active interference items.
    n_filler : int, optional
        Override for active filler items.
    show_break : bool
        Whether break items are shown in the SPC.
    show_fillers : bool
        Whether filler items are shown in the SPC.

    Returns
    -------
    int

    """
    nb = n_break if n_break is not None else paradigm.n_break
    ni = n_interference if n_interference is not None else paradigm.n_interference
    nf = n_filler if n_filler is not None else paradigm.n_filler
    return (
        paradigm.n_film
        + (nb if show_break else 0)
        + ni
        + (nf if show_fillers else 0)
    )
