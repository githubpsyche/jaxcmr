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

# %% Imports

from jaxcmr.helpers import (
    Integer, Float, Array, Bool, ScalarFloat, ScalarInteger, ScalarBool, recall_outcomes, lb
)
from plum import dispatch
from typing import Any
from jax import jit, lax, numpy as jnp
from jaxcmr.memorysearch import MemorySearch
from jaxcmr.memory import OneWayMemory, probe, associate
from jaxcmr.context import (
    Context,
    integrate,
    integrate_start_context,
    integrate_delay_context,
)
from jaxcmr.helpers import replace
from functools import partial

# %% Public Interface

__all__ = [
    "CMR",
    "experience_item",
    "stop_probability",
    "item_probability",
    "outcome_probabilities",
    "outcome_probability",
    "start_retrieving",
    "retrieve_item",
]


# %% Types


class CMR(MemorySearch, mutable=True):
    mfc: OneWayMemory
    mcf: OneWayMemory
    context: Context
    encoding_drift_rate: ScalarFloat
    delay_drift_rate: ScalarFloat
    start_drift_rate: ScalarFloat
    recall_drift_rate: ScalarFloat
    mfc_learning_rate: ScalarFloat
    mcf_learning_rate: ScalarFloat
    stop_probability_scale: ScalarFloat
    stop_probability_growth: ScalarFloat
    is_active: bool | ScalarBool
    item_count: ScalarInteger
    items: Integer[Array, "item_count item_features"]
    encoding_index: ScalarInteger
    recall_total: ScalarInteger
    recall_sequence: Integer[Array, "item_count"]
    recall_mask: bool | Bool[Array, "item_count"]

# %% Encoding


@partial(jit, static_argnums=(1,))
@dispatch
def experience_item(model: CMR, item_index: ScalarInteger) -> CMR:
    """Experience a study item at the specified index (1-indexed)"""
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


# %% Event Probabilities

@dispatch
def stop_probability(
        stop_probability_scale: ScalarFloat,
        stop_probability_growth: ScalarFloat,
        recall_total: ScalarInteger,
):
    """Probability of stopping recall given total number of items recalled and model parameters"""
    return stop_probability_scale * jnp.exp(recall_total * stop_probability_growth)


@dispatch
def stop_probability(model: CMR, _: Any = None) -> ScalarFloat:
    """Probability of stopping recall given the current state of the model"""
    return lax.cond(
        model.is_active,
        lambda _: lax.min(
            stop_probability(
                model.stop_probability_scale, model.stop_probability_growth, model.recall_total),
            1.0 - ((model.item_count - model.recall_total) * lb)
        ),
        lambda _: 1.,
        None,
    )


@dispatch
def item_probability(
        model: CMR, choice: ScalarInteger) -> ScalarFloat:
    """Probability of retrieving the item with the specified index (1-indexed)"""
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation * (1 - model.recall_mask)  # mask recalled items
    return (1 - p_stop) * (item_activation[choice - 1] / jnp.sum(item_activation))


@dispatch
def outcome_probabilities(model: CMR) -> Float[Array, "recall_outcomes"]:
    """Return the probability of each possible retrieval outcome, with termination indexed at 0"""
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation * (1 - model.recall_mask)  # mask recalled items
    return jnp.hstack(
        (p_stop, ((1 - p_stop) * item_activation / jnp.sum(item_activation)))
    )


@dispatch
def outcome_probability(
        model: CMR, choice: ScalarInteger
) -> ScalarFloat:
    """Return the probability of a particular retrieval outcome"""
    return lax.cond(
        choice > 0,
        lambda _: item_probability(model, choice),
        lambda _: stop_probability(model),
        None,
    )


# %% Item Retrieval

@dispatch
def start_retrieving(model: CMR) -> CMR:
    """Evolve model reflect its initial state at the start of free recall"""
    new_context = integrate_delay_context(model.context, model.delay_drift_rate)
    new_context = integrate_start_context(new_context, model.start_drift_rate)
    return replace(model, context=new_context)


@dispatch
def retrieve_item(model: CMR, choice: ScalarInteger) -> CMR:
    """Retrieve item with index choice-1"""
    context_input = probe(model.mfc, model.items[choice - 1])
    return replace(
        model,
        context=integrate(model.context, context_input, model.recall_drift_rate),
        recall_sequence=model.recall_sequence.at[model.recall_total].set(choice - 1),
        recall_mask=model.recall_mask.at[choice - 1].set(True),
        recall_total=model.recall_total + 1,
    )
