""" 
Outcome probability functions for memory search models.

Memory search involves encoding items into memory and eventually performing a retrieval operation to recall an item from memory.

The outcome probability functions here are used to compute the likelihood of a particular retrieval outcome given a state of the model.
"""

# %% Imports
from plum import dispatch
from jaxcmr.memorysearch.types import CMR
from jaxcmr.memory import probe
from jax import jit, lax, numpy as jnp
from beartype.typing import Any
from jaxcmr.helpers import ScalarFloat, Float, Array, ScalarInteger, lb, recall_outcomes

# %% Exports

__all__ = [
    "outcome_probability",
    "stop_probability",
    "item_probability",
]

# %% Base outcome_probability functions


@jit
@dispatch
def outcome_probability(model: CMR) -> Float[Array, "recall_outcomes"]:
    """The probability distribution over possible retrieval outcomes; termination indexed at 0"""
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation * (1 - model.recall_mask)  # mask recalled items
    item_activation_sum = jnp.sum(item_activation)
    return jnp.hstack(
        (p_stop, ((1 - p_stop) * item_activation / lax.select(item_activation_sum == 0, 1., item_activation_sum)))
    )


@jit
@dispatch
def outcome_probability(model: CMR, choice: ScalarInteger) -> ScalarFloat:
    """Return the probability of a particular retrieval outcome"""
    return lax.cond(
        choice > 0,
        lambda _: item_probability(model, choice),
        lambda _: stop_probability(model),
        None,
    )


# %% Helper Functions


@jit
@dispatch
def stop_probability(
    stop_probability_scale: ScalarFloat,
    stop_probability_growth: ScalarFloat,
    recall_total: ScalarInteger,
) -> ScalarFloat:
    """Probability of stopping recall given total number of items recalled and model parameters"""
    return stop_probability_scale * jnp.exp(recall_total * stop_probability_growth)


@jit
@dispatch
def stop_probability(model: CMR, _: Any = None) -> ScalarFloat:
    """
    Probability of stopping recall given the current state of the model

    Configured so that unrecalled items always have a non-zero probability of being recalled.
    """
    return lax.cond(
        model.is_active,
        lambda _: lax.min(
            stop_probability(
                model.stop_probability_scale,
                model.stop_probability_growth,
                model.recall_total,
            ),
            1.0 - (sum(1-model.recall_mask) * lb),
        ),
        lambda _: 1.0,
        None,
    )


@jit
@dispatch
def item_probability(model: CMR, choice: ScalarInteger) -> ScalarFloat:
    """Probability of retrieving the item with the specified index (1-indexed)"""
    p_stop = stop_probability(model)
    item_activation = probe(model.mcf, model.context.state) + lb
    item_activation = item_activation * (1 - model.recall_mask)  # mask recalled items
    return (1 - p_stop) * (item_activation[choice - 1] / jnp.sum(item_activation))
