"""
Linear Associative Memory
"""

from jaxtyping import Integer, Float, Array
from flax.struct import PyTreeNode
import jax.numpy as jnp
from plum import dispatch
from jax import lax


#%% Type Hierarchy

class LinearAssociativeMemory(PyTreeNode):
    pass

class LinearAssociativeMfc(LinearAssociativeMemory):
    state: Float[Array, "item_features context_features"]

class LinearAssociativeMcf(LinearAssociativeMemory):
    state: Float[Array, """context_features item_features"""]

#%% Mfc Initialization

@dispatch
def initialize_mfc(
    item_count: int | Integer[Array, ""],
    learning_rate: float | Float[Array, ""]
    ) -> LinearAssociativeMfc:
    "Initialize a linear associative feature-to-context memory"
    memory = jnp.eye(item_count, item_count + 2)
    memory = jnp.hstack([jnp.zeros((item_count, 1)), memory[:, :-1]])
    memory = memory * (1 - learning_rate)
    return LinearAssociativeMfc(memory)


@dispatch
def initialize_mfc(
    items: Float[Array, "item_count item_features"], 
    learning_rate: float | Float[Array, ""], 
    ) -> LinearAssociativeMfc:
    "Generalized initialize function for LinearAssociativeMfc with arbitrary item representations"
    item_count = items.shape[0]
    item_feature_count = items.shape[1]
    context_feature_count = item_count + 2

    contexts = jnp.eye(item_count, context_feature_count, 1)
    memory = jnp.zeros((item_feature_count, context_feature_count))
    memory = lax.fori_loop(
        0, item_count, lambda i, memory: linear_associate(memory, 1-learning_rate, items[i], contexts[i]), memory)

    return LinearAssociativeMfc(memory)

#%% Mcf Initialization

@dispatch
def initialize_mcf(
    item_count: int | Integer[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""]
    ) -> LinearAssociativeMcf:
    "Initialize a linear associative context-to-feature memory"
    memory = jnp.full((item_count, item_count), shared_support)
    memory = memory.at[jnp.diag_indices(item_count)].set(item_support)
    memory = jnp.vstack(
        (jnp.zeros((1, item_count)), memory, jnp.zeros((1, item_count))))
    return LinearAssociativeMcf(memory)


@dispatch
def initialize_mcf(
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
    memory = lax.fori_loop(
        0, item_count, lambda i, memory: linear_associate(memory, 1., contexts[i], items[i]), memory)
    
    return LinearAssociativeMcf(memory)

#%% 

@dispatch
def linear_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"]
    ) -> Float[Array, "input_features output_features"]:
    "Associate input and output feature patterns in a linear associative memory state"
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern))


@dispatch
def associate(
    memory: LinearAssociativeMemory,
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"]
    ) -> LinearAssociativeMemory:
    "Associate input and output feature patterns in a linear associative memory"
    return memory.replace(state=linear_associate(
        memory.state, learning_rate, input_feature_pattern, output_feature_pattern
        ))