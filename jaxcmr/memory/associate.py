"""
Associate functions for one-way associative memories.

One-way associative memories are used to associate input and output feature vectors.
"""

# %% Imports

from plum import dispatch
from jax import jit, numpy as jnp
from jaxcmr.memory.types import (
    OneWayMemory,
    LinearAssociativeMemory,
    InstanceMemory,
    LinearAssociativeMcf,
    InstanceMcf,
    LinearAssociativeMfc,
)
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarFloat,
    ScalarInteger,
    input_features,
    output_features,
    replace,
)

# %% Exports

__all__ = ["associate", "hebbian_associate", "instance_associate"]


# %% Abstract Asspcoate Function


@dispatch.abstract
def associate(
    memory: OneWayMemory,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "output_features"]:
    """Associate input and output feature patterns in a one-way associative memory"""


# %% Associative Functions for Linear Associative Memories


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


@jit
@dispatch
def associate(
    memory: LinearAssociativeMemory,
    learning_rate: ScalarFloat,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> LinearAssociativeMemory:
    """Associate input and output feature patterns in a linear associative memory"""
    return replace(
        memory,
        state=hebbian_associate(
            memory.state, learning_rate, input_feature_pattern, output_feature_pattern
        ),
    )


# %% Associative Functions for Instance Memories


@jit
@dispatch
def instance_associate(
    memory_state: Float[Array, "instances instance_features"],
    encoding_index: ScalarInteger,
    learning_rate: ScalarFloat,
    input_pattern: Float[Array, "input_features"],
    output_pattern: Float[Array, "output_features"],
) -> Float[Array, "instances instance_features"]:
    """Associate two patterns in a instance-based memory state"""
    return memory_state.at[encoding_index].set(
        jnp.concatenate((input_pattern, output_pattern * learning_rate))
    )


@jit
@dispatch
def associate(
    memory: InstanceMemory,
    learning_rate: ScalarFloat,
    input_pattern: Float[Array, "input_features"],
    output_pattern: Float[Array, "output_features"],
) -> InstanceMemory:
    """Associate input and output feature patterns in a instance-based memory"""
    return replace(
        memory,
        state=instance_associate(
            memory.state,
            memory.encoding_index,
            learning_rate,
            input_pattern,
            output_pattern,
        ),
        encoding_index=memory.encoding_index + 1,
    )
