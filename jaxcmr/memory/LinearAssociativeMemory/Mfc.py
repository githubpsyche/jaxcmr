"""
Linear Associative $M^{FC}$

Initialization functions for a feature-to-context linear associative memory as specified for CMR.
"""

#%% Imports

from jaxtyping import Integer, Float, Array
from plum import dispatch
from jax import lax, jit, numpy as jnp
from functools import partial
from jaxcmr.memory.LinearAssociativeMemory.LinearAssociativeMemory import (
    LinearAssociativeMemory, 
    hebbian_associate,
)

#%% Public interface

__all__ = ['init_linear_mfc']

#%% Initialization

@partial(jit, static_argnums=(0,))
def basic_init_mfc(
    item_count: int | Integer[Array, ""],
    learning_rate: float | Float[Array, ""]
    ) -> LinearAssociativeMemory:
    "A linear associative feature-to-context memory assuming one-hot item representations"
    memory = jnp.eye(item_count, item_count + 2)
    memory = jnp.hstack([jnp.zeros((item_count, 1)), memory[:, :-1]])
    memory = memory * (1 - learning_rate)
    return LinearAssociativeMemory(memory)


@jit
def generalized_init_mfc(
    items: Float[Array, "item_count item_features"], 
    learning_rate: float | Float[Array, ""], 
    ) -> LinearAssociativeMemory:

    "A linear associative feature-to-context memory from arbitrary item representations"
    item_count = items.shape[0]
    item_feature_count = items.shape[1]
    context_feature_count = item_count + 2
    contexts = jnp.eye(item_count, context_feature_count, 1)
    memory = jnp.zeros((item_feature_count, context_feature_count))
    return LinearAssociativeMemory(lax.fori_loop(
        0, 
        item_count, 
        lambda i, memory: hebbian_associate(memory, 1-learning_rate, items[i], contexts[i]), 
        memory
        ))
        

@dispatch
def init_linear_mfc(
    item_count: int | Integer[Array, ""], 
    learning_rate: float | Float[Array, ""]
    ) -> LinearAssociativeMemory:
    return basic_init_mfc(item_count, learning_rate)


@dispatch
def init_linear_mfc(
    items: Float[Array, "item_count item_features"], 
    learning_rate: float | Float[Array, ""]
    ) -> LinearAssociativeMemory:
    return generalized_init_mfc(items, learning_rate)