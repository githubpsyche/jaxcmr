""" 
Retrieve functions for memory search models.

Memory search involves encoding items into memory and eventually performing a retrieval operation to recall an item from memory.
"""

# %% Imports
from plum import dispatch
from jaxcmr.memorysearch.types import MemorySearch, CMR
from jaxcmr.memory import probe
from jax import jit, lax
from jaxcmr.context import integrate, integrate_outlist_context, integrate_start_context
from jaxcmr.helpers import (
    ScalarInteger,
    replace,
)

# %% Exports

__all__ = ["start_retrieving", "stop_recall", "retrieve", "_retrieve_item"]

# %% Base retrieval functions


@jit
@dispatch
def retrieve(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Perform retrieval event, either item recall (choice > 0) or termination (choice = 0)"""
    return lax.cond(choice > 0, _retrieve_item, stop_recall, model, choice)


@dispatch.abstract
def _retrieve_item(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Retrieve an item from memory"""


@dispatch
def _retrieve_item(model: CMR, choice: ScalarInteger) -> CMR:
    """Retrieve item with index choice-1"""
    context_input = probe(model.mfc, model.items[choice - 1])
    return replace(
        model,
        context=integrate(model.context, context_input, model.recall_drift_rate),
        recall_sequence=model.recall_sequence.at[model.recall_total].set(choice - 1),
        recall_mask=model.recall_mask.at[choice - 1].set(True),
        recall_total=model.recall_total + 1,
    )


# %% Terminating retrieval


@dispatch
def stop_recall(model: MemorySearch, _: ScalarInteger = 0) -> MemorySearch:
    """The model shifts to inactive mode and will not retrieve any more items"""
    return replace(model, is_active=False)


# %% Initiating retrieval phase


@dispatch.abstract
def start_retrieving(model: MemorySearch) -> MemorySearch:
    """Evolve model reflect its initial state at the start of free recall"""


@dispatch
def start_retrieving(model: CMR) -> CMR:
    """Evolve model reflect its initial state at the start of free recall"""
    new_context = integrate_outlist_context(model.context, model.delay_drift_rate)
    new_context = integrate_start_context(new_context, model.start_drift_rate)
    return replace(model, context=new_context)
