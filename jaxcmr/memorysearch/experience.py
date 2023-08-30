"""
Experience functions for memory search models.

Memory search involves encoding items into memory and eventually performing a retrieval operation to recall an item from memory.
"""

# %% Imports
from plum import dispatch
from functools import partial
from jaxcmr.memorysearch.types import MemorySearch, CMR
from jaxcmr.memory import probe, associate
from jax import jit, lax
from jaxcmr.context import integrate
from jaxcmr.helpers import (
    Integer,
    Array,
    ScalarInteger,
    study_events,
    replace,
)

# %% Exports

__all__ = ["experience"]

# %% A base experience function encodes a single item into memory based on its index


@dispatch
def experience(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Experience a study item at the specified index (1-indexed) or ignore it (choice = 0)"""
    return lax.cond(
        choice == 0,
        lambda _: model,
        lambda _: _experience_item(model, choice - 1),
        None,
    )


@dispatch.abstract
def _experience_item(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Experience a study item at the specified index (0-indexed)"""


@partial(jit, static_argnums=(1,))
@dispatch
def _experience_item(model: CMR, item_index: ScalarInteger) -> CMR:
    """Experience a study item at the specified index (0-indexed)"""
    encoded_item = model.items[item_index]
    context_input = probe(model.mfc, encoded_item)
    new_context = integrate(model.context, context_input, model.encoding_drift_rate)
    return replace(
        model,
        context=new_context,
        mfc=associate(
            model.mfc, model.mfc_learning_rate, encoded_item, new_context.state
        ),
        mcf=associate(
            model.mcf, model.mcf_learning_rate, new_context.state, encoded_item
        ),
        encoding_index=model.encoding_index + 1,
    )


# %% Multi-item experience functions


@dispatch
def experience(model: MemorySearch) -> MemorySearch:
    """Experience all study items initialized with the model in order of index."""
    return lax.fori_loop(1, model.item_count + 1, lambda i, m: experience(m, i), model)


@dispatch
def experience(
    model: MemorySearch, choices: Integer[Array, "study_events"]
) -> MemorySearch:
    """Experience study items initialized with the model in the specified order of indices."""
    return lax.fori_loop(
        0, choices.shape[0], lambda i, m: experience(m, choices[i]), model
    )
