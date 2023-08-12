"""
Linear Associative $M^{CF}$

Initialization functions for a context-to-feature linear associative memory as specified for CMR.
"""

# %% Imports

from jaxtyping import Integer, Float, Array
from plum import dispatch
from jax import lax, jit, numpy as jnp
from functools import partial
from jaxcmr.memory.LinearAssociativeMemory.LinearAssociativeMemory import (
    LinearAssociativeMemory,
    hebbian_associate,
    scale_activation,
)

# %% Public interface

__all__ = [
    "LinearAssociativeMcf",
    "probe",
]

# %% Subtype of LinearAssociativeMemory


class LinearAssociativeMcf(LinearAssociativeMemory, mutable=True):
    @dispatch
    def __init__(
        self,
        state: Float[Array, "context_features item_features"],
        choice_sensitivity: float | Float[Array, ""] = 1.0,
    ):
        self.state = state
        self.choice_sensitivity = choice_sensitivity

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: int | Integer[Array, ""],
        shared_support: float | Float[Array, ""],
        item_support: float | Float[Array, ""],
        choice_sensitivity: float | Float[Array, ""] = 1.0,
    ):
        return cls(
            basic_init_linear_mcf(item_count, shared_support, item_support),
            choice_sensitivity,
        )

    @classmethod
    @dispatch
    def create(
        cls,
        items: Float[Array, "item_count item_features"],
        shared_support: float | Float[Array, ""],
        item_support: float | Float[Array, ""],
        choice_sensitivity: float | Float[Array, ""] = 1.0,
    ):
        return cls(
            generalized_init_linear_mcf(items, shared_support, item_support),
            choice_sensitivity,
        )


# %% Initialization


@partial(jit, static_argnums=(0,))
def basic_init_linear_mcf(
    item_count: int | Integer[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""],
) -> Float[Array, "context_features item_features"]:
    "Initialize a linear associative context-to-feature memory"
    memory = jnp.full((item_count, item_count), shared_support)
    memory = memory.at[jnp.diag_indices(item_count)].set(item_support)
    return jnp.vstack((jnp.zeros((1, item_count)), memory, jnp.zeros((1, item_count))))


@jit
def generalized_init_linear_mcf(
    items: Float[Array, "item_count item_features"],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""],
) -> Float[Array, "context_features item_features"]:
    "Generalized initialize function for LinearAssociativeMcf with arbitrary item representations"
    item_count = items.shape[0]
    item_feature_count = items.shape[1]
    context_feature_count = item_count + 2

    contexts = jnp.eye(item_count, context_feature_count, 1)
    memory = jnp.zeros((context_feature_count, item_feature_count))
    items = items * item_support
    items = items + (
        shared_support - jnp.eye(item_count, item_feature_count) * shared_support
    )
    return lax.fori_loop(
        0,
        item_count,
        lambda i, memory: hebbian_associate(memory, 1.0, contexts[i], items[i]),
        memory,
    )


# %% Probe


@jit
@dispatch
def probe(
    memory: LinearAssociativeMcf, probe: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    "Return the activation vector of a linear associative memory"
    return scale_activation(jnp.dot(probe, memory.state), memory.choice_sensitivity)
