"""
Linear Associative $M^{CF}$

Initialization functions for a context-to-feature linear associative memory as specified for CMR. Item
representations and a space of one-hot context states are associated according to the Hebbian learning rule based on
a provided learning rate.

When initialized with an item count, items are represented as one-hot vectors.
Otherwise, items can be explicitly provided as a matrix of item vectors.
"""

# %% Imports
from jaxcmr.memory.types.OneWayMemory import LinearAssociativeMemory
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarInteger,
    ScalarFloat,
    input_features,
    output_features,
)
from plum import dispatch
from jax import lax, jit, numpy as jnp
from functools import partial

# %% Exports

__all__ = ["LinearAssociativeMcf"]


# %% Base Type for Linear Associative Mcf


class LinearAssociativeMcf(LinearAssociativeMemory, mutable=True):
    @dispatch
    def __init__(
        self,
        state: Float[Array, "context_features item_features"],
        choice_sensitivity: ScalarFloat = 1.0,
    ):
        self.state = state
        self.choice_sensitivity = choice_sensitivity

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: ScalarInteger,
        shared_support: ScalarFloat,
        item_support: ScalarFloat,
        choice_sensitivity: ScalarFloat = 1.0,
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
        shared_support: ScalarFloat,
        item_support: ScalarFloat,
        choice_sensitivity: ScalarFloat = 1.0,
    ):
        return cls(
            generalized_init_linear_mcf(items, shared_support, item_support),
            choice_sensitivity,
        )

    @property
    def input_features(self):
        return self.state.shape[0]

    @property
    def output_features(self):
        return self.state.shape[1]


# %% Type-Specific Helper Functions


@partial(jit, static_argnums=(0,))
def basic_init_linear_mcf(
    item_count: ScalarInteger,
    shared_support: ScalarFloat,
    item_support: ScalarFloat,
) -> Float[Array, "context_features item_features"]:
    """Initialize a linear associative context-to-feature memory"""
    memory = jnp.full((item_count, item_count), shared_support)
    memory = memory.at[jnp.diag_indices(item_count)].set(item_support)
    return jnp.vstack((jnp.zeros((1, item_count)), memory, jnp.zeros((1, item_count))))


@jit
def generalized_init_linear_mcf(
    items: Float[Array, "item_count item_features"],
    shared_support: ScalarFloat,
    item_support: ScalarFloat,
) -> Float[Array, "context_features item_features"]:
    """Initialize LinearAssociativeMcf with arbitrary item representations"""
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
        lambda i, m: hebbian_associate(m, 1.0, contexts[i], items[i]),
        memory,
    )


@jit
@dispatch
def hebbian_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: ScalarFloat,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "input_features output_features"]:
    """Associate input and output feature patterns in a M x N linear associative memory state"""
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern)
    )
