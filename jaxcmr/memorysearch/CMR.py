"""
The Context Maintenance and Retrieval (CMR) model of memory search.
Concrete type and functions implementing the CMR model as specified by Morton & 
Polyn (2016).

Three major components define how the CMR model implements memory search:
- A context representation
- A feature-to-context associative memory
- A context-to-feature associative memory

Following principles of behavioral over implementational inheritance,
as long as subtypes are supported by available implementations of basic accessor 
and getter functions, all other functions will work, regardless of subtype 
implementation.
"""

#%% Imports

from jaxtyping import Integer, Float, Array, Bool
from plum import dispatch
from typing import Any
from jax import jit, lax, numpy as jnp
from jaxcmr.memorysearch import MemorySearch
from jaxcmr.memory import OneWayMemory, probe, associate
from jaxcmr.context import Context, integrate, integrate_start_context, integrate_delay_context
from jaxcmr.helpers import replace
from functools import partial
lb = jnp.finfo(float).eps

#%% Public Interface

__all__ = [
    'CMR',
    'experience_item',
    'stop_probability',
    'item_probability',
    'outcome_probabilities',
    'outcome_probability',
    'start_retrieving',
    'retrieve_item',
]

#%% Types

class CMR(MemorySearch, mutable=True):

    def __init__(
        self,
        mfc: OneWayMemory,
        mcf: OneWayMemory,
        context: Context,
        encoding_drift_rate: float | Float[Array, ""],
        delay_drift_rate: float | Float[Array, ""],
        start_drift_rate: float | Float[Array, ""],
        recall_drift_rate: float | Float[Array, ""],
        mfc_learning_rate: float | Float[Array, ""],
        mcf_learning_rate: float | Float[Array, ""],
        stop_probability_scale: float | Float[Array, ""],
        stop_probability_growth: float | Float[Array, ""],
        is_active: bool | Bool[Array, ""],
        item_count: int | Integer[Array, ""],
        items: Integer[Array, "item_count item_features"],
        encoding_index: int | Integer[Array, ""],
        recall_total: int | Integer[Array, ""],
        recall_sequence: Integer[Array, "item_count"],
        recall_mask: bool | Bool[Array, "item_count"],
    ):
        raise NotImplementedError


#%% Encoding

@partial(jit, static_argnums=(1,))
@dispatch
def experience_item(model: CMR, item_index: int | Integer[Array, ""]) -> CMR:
    "Experience a study item at the specified index (1-indexed)"
    encoded_item = model.items[item_index]
    context_input = probe(model.mfc, encoded_item)
    new_context = integrate(model.context, context_input, model.encoding_drift_rate)
    return replace(
        model,
        context=new_context, 
        mfc=associate(model.mfc, model.mfc_learning_rate, encoded_item, new_context.state), 
        mcf=associate(model.mcf, model.mcf_learning_rate, new_context.state, encoded_item), 
        encoding_index=model.encoding_index+1
    )

#%% Event Probabilities

@dispatch
def stop_probability(model: CMR, _: Any = None) -> Float[Array, ""]:
    "Probability of stopping recall given the current state of the model"
    return lax.min(
        jnp.exp(model.recall_total * model.stop_probability_growth) * model.stop_probability_scale,
        1.0 - ((model.item_count - model.recall_total) * lb)
    )

@dispatch
def item_probability(model: CMR, choice: int | Integer[Array, ""]) -> Float[Array, ""]:
    "Probability of retrieving the item with the specified index (1-indexed)"
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation * (1 - model.recall_mask) # mask recalled items
    return (1 - p_stop) * (item_activation[choice-1] / jnp.sum(item_activation))

@dispatch
def outcome_probabilities(model: CMR) -> Float[Array, "outcome_count"]:
    "Return the probability of each possible retrieval outcome, with termination indexed at 0"
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation.at[:].multiply(1 - model.recall_mask) # mask recalled items
    return jnp.hstack((p_stop, (((1 - p_stop) * item_activation) / jnp.sum(item_activation))))

@dispatch
def outcome_probability(model: CMR, choice: int | Integer[Array, ""]) -> Float[Array, ""]:
    "Return the probability of a particular retrieval outcome"
    return lax.cond(
        choice > 0, 
        lambda _: item_probability(model, choice), 
        lambda _: stop_probability(model), 
        None)

#%% Item Retrieval

@dispatch
def start_retrieving(model: CMR) -> CMR:
    "Evolve model reflect its initial state at the start of free recall"
    new_context = integrate_delay_context(model.context, model.delay_drift_rate)
    new_context = integrate_start_context(new_context, model.start_drift_rate)
    return replace(model, context=new_context)

@dispatch
def retrieve_item(model: CMR, choice: int | Integer[Array, ""]) -> CMR:
    "Retrieve item with index choice-1"
    context_input = probe(model.mfc, model.items[choice-1])
    return replace(
        model,
        context = integrate(model.context, context_input, model.recall_drift_rate),
        recall_sequence = model.recall_sequence.at[model.recall_total].set(choice-1),
        recall_mask = model.recall_mask.at[choice-1].set(True),
        recall_total = model.recall_total + 1
    )