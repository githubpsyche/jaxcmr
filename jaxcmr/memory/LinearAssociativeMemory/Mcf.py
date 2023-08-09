"""
Linear Associative $M^{CF}$

Initialization functions for a context-to-feature linear associative memory as specified for CMR.
"""

#%% Imports

from jaxtyping import Integer, Float, Array
from plum import dispatch
from jax import lax, jit, numpy as jnp
from functools import partial
from jaxcmr.helpers import replace
from simple_pytree import dataclass
from jaxcmr.memory.LinearAssociativeMemory.LinearAssociativeMemory import (
    LinearAssociativeMemory, 
    hebbian_associate,
)

#%% Public interface

__all__ = [
    'LinearAssociativeMcf',
    'init_linear_mcf',
    ]

#%% Subtype of LinearAssociativeMemory

@dataclass
class LinearAssociativeMcf(LinearAssociativeMemory, mutable=True):
    state: Float[Array, "context_features item_features"]

#%% Initialization

@partial(jit, static_argnums=(0,))
def basic_init_linear_mcf(
    item_count: int | Integer[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""]
    ) -> LinearAssociativeMemory:
    "Initialize a linear associative context-to-feature memory"
    memory = jnp.full((item_count, item_count), shared_support)
    memory = memory.at[jnp.diag_indices(item_count)].set(item_support)
    memory = jnp.vstack(
        (jnp.zeros((1, item_count)), memory, jnp.zeros((1, item_count))))
    return LinearAssociativeMcf(memory)


@jit
def generalized_init_linear_mcf(
    items: Float[Array, "item_count item_features"], 
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""]
    ) -> LinearAssociativeMcf:
    "Generalized initialize function for LinearAssociativeMcf with arbitrary item representations"
    item_count = items.shape[0]
    item_feature_count = items.shape[1]
    context_feature_count = item_count + 2

    contexts = jnp.eye(item_count, context_feature_count, 1)
    memory = jnp.zeros((context_feature_count, item_feature_count))
    items = items * item_support
    items = items + (shared_support - jnp.eye(item_count, item_feature_count) * shared_support)
    return LinearAssociativeMcf(
        lax.fori_loop(
            0, 
            item_count, 
            lambda i, memory: hebbian_associate(memory, 1., contexts[i], items[i]), 
            memory
        ))


@dispatch
def init_linear_mcf(
    item_count: int | Integer[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""]
    ) -> LinearAssociativeMcf:
    return basic_init_linear_mcf(item_count, shared_support, item_support)


@dispatch
def init_linear_mcf(
    items: Float[Array, "item_count item_features"], 
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""]
    ) -> LinearAssociativeMcf:
    return generalized_init_linear_mcf(items, shared_support, item_support)
