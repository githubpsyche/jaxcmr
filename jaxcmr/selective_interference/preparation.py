"""Model preparation for selective interference simulations.

Before running a parameter sweep, each simulated subject needs a CMR
model that has already encoded the film and break phases and received
a context-reinstatement reminder.  This "cached" model state is
expensive to build but identical across sweep conditions, so it is
computed once and reused.

``prepare_single_subject`` takes one subject's fitted parameters
(including phase-configuration scales), encodes the film items,
encodes the break items, reinstates start-of-list context, and runs
the reminder sequence.  It returns the model ready for interference
encoding.

``prepare_all_subjects`` broadcasts scalar phase-configuration
scales into the per-subject parameter dict, then applies
``prepare_single_subject`` across all subjects in parallel via
``jit(vmap(...))``.

"""

from typing import Mapping

import jax.numpy as jnp
from jax import jit, lax, vmap

from jaxcmr.typing import Array, Float, Float_, Integer

from .typing import PhasedMemorySearch, PhasedMemorySearchCreateFn


def prepare_single_subject(
    params: Mapping[str, Float_],
    film_items: Integer[Array, " n_film"],
    break_items: Integer[Array, " n_break"],
    list_length: int,
    model_factory: PhasedMemorySearchCreateFn,
) -> PhasedMemorySearch:
    """Create model, encode film and break, run reminder.

    Parameters
    ----------
    params : Mapping[str, Float_]
        Single-subject parameters including phase-specific rates
        (``break_drift_rate``, ``break_mcf_scale``,
        ``reminder_start_drift_rate``, ``reminder_drift_rate``).
    film_items : Integer[Array, " n_film"]
        Item IDs for the film phase.
    break_items : Integer[Array, " n_break"]
        Item IDs for the break phase.
    list_length : int
        Total list length (determines matrix sizes).
    model_factory : PhasedMemorySearchCreateFn
        Creator function for PhasedMemorySearch models.

    Returns
    -------
    PhasedMemorySearch

    """
    model = model_factory(list_length, params, None)

    # Film encoding
    model = lax.fori_loop(
        0, film_items.size, lambda i, m: m.experience_film(film_items[i]), model
    )

    # Break encoding
    model = lax.fori_loop(
        0, break_items.size, lambda i, m: m.experience_break(break_items[i]), model
    )

    # Start-of-list reinstatement + reminder
    model = model.start_reminders()
    model = lax.fori_loop(
        0, film_items.size, lambda i, m: m.remind(film_items[i]), model
    )
    return model


def _prepare_all_subjects(
    params: Mapping[str, Float[Array, " n_subjects"]],
    film_items: Integer[Array, " n_film"],
    break_items: Integer[Array, " n_break"],
    break_drift_scale: Float_,
    break_mcf_scale: Float_,
    reminder_start_drift_scale: Float_,
    reminder_drift_scale: Float_,
    list_length: int,
    model_factory: PhasedMemorySearchCreateFn,
) -> PhasedMemorySearch:
    """Configure phase scales and prepare all subjects in parallel.

    Parameters
    ----------
    params : Mapping[str, Float[Array, " n_subjects"]]
        Per-subject fitted parameters.
    film_items : Integer[Array, " n_film"]
        Item IDs for the film phase.
    break_items : Integer[Array, " n_break"]
        Item IDs for the break phase.
    break_drift_scale : Float_
        Multiplier on encoding drift rate during break.
    break_mcf_scale : Float_
        Multiplier on MCF learning rate during break.
    reminder_start_drift_scale : Float_
        Multiplier on start drift rate for pre-reminder reinstatement.
    reminder_drift_scale : Float_
        Multiplier on encoding drift rate for reminder presentation.
    list_length : int
        Total list length (determines matrix sizes).
    model_factory : PhasedMemorySearchCreateFn
        Creator function for PhasedMemorySearch models.

    Returns
    -------
    PhasedMemorySearch

    """
    encoding_drift = params["encoding_drift_rate"]
    start_drift = params["start_drift_rate"]
    n = encoding_drift.shape[0]
    phased_params = {
        **params,
        "break_drift_rate": jnp.clip(break_drift_scale * encoding_drift, 0.0, 1.0),
        "break_mcf_scale": jnp.full(n, break_mcf_scale),
        "reminder_start_drift_rate": jnp.clip(
            reminder_start_drift_scale * start_drift, 0.0, 1.0
        ),
        "reminder_drift_rate": jnp.clip(
            reminder_drift_scale * encoding_drift, 0.0, 1.0
        ),
    }
    return vmap(
        prepare_single_subject,
        in_axes=(0, None, None, None, None),
    )(
        phased_params,
        film_items,
        break_items,
        list_length,
        model_factory,
    )


prepare_all_subjects = jit(
    _prepare_all_subjects,
    static_argnums=(7, 8),
)
