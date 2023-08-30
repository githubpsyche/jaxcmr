"""
Probe functions for one-way associative memories.

One-way associative memories are used to associate input and output feature patterns.
To retrieve the output vector associated with an input vector, use the probe function.
"""

# %% Imports

from plum import dispatch
from jax import jit, numpy as jnp
from jaxcmr.memory.types import (
    OneWayMemory,
    LinearAssociativeMemory,
    LinearAssociativeMcf,
    InstanceMcf,
)
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarFloat,
    input_features,
    output_features,
    instances,
    power_scale,
)

# %% Exports

__all__ = ["probe", "linear_probe", "instance_probe", "trace_activation"]


# %% Abstract Probe Function


@dispatch.abstract
def probe(
    memory: OneWayMemory, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Retrieve the output associated with an input in a one-way associative memory"""


# %% Probe Functions for Linear Associative Memories

"""
Probing a linear associative memory involves a dot product between the input feature pattern and the memory state 
matrix.

Depending on the type of linear associative memory, the activation vector is scaled by a power factor and/or 
normalized to unit magnitude.

When typing constrains whether power scaling or normalization is required, we dispatch type-specialized probe 
functions to avoid unnecessary computation.
"""


@jit
@dispatch
def linear_probe(
    memory_state: Float[Array, "input_features output_features"],
    input_feature_pattern: Float[Array, "input_features"],
) -> Float[Array, "output_features"]:
    """Retrieve the output associated with an input in an M x N linear associative memory state"""
    return jnp.dot(input_feature_pattern, memory_state)


@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory,
    input_feature_pattern: Float[Array, "input_features"],
) -> Float[Array, "output_features"]:
    """Retrieve the output associated with an input in a LinearAssociativeMemory"""
    return linear_probe(memory.state, input_feature_pattern)


@jit
@dispatch
def probe(
    memory: LinearAssociativeMcf, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Retrieve the output associated with an input in a LinearAssociativeMcf"""
    base_activation = linear_probe(memory.state, input_feature_pattern)
    return power_scale(base_activation, memory.choice_sensitivity)


# %% Probe Functions for Instance-Based Memories


@jit
@dispatch
def trace_activation(
    state: Float[Array, "instances instance_features"],
    input_feature_pattern: Float[Array, "input_features"],
    trace_scale: ScalarFloat,
) -> Float[Array, "instances"]:
    """Retrieve the output associated with an input in a  instance-based memory state"""
    relevant_state = state[:, : input_feature_pattern.shape[0]]
    a = jnp.dot(relevant_state, input_feature_pattern)
    return power_scale(a, trace_scale)


@jit
@dispatch
def instance_probe(
    memory_state: Float[Array, "instances instance_features"],
    input_feature_pattern: Float[Array, "input_features"],
    feature_scale: ScalarFloat,
    trace_scale: ScalarFloat,
) -> Float[Array, "output_features"]:
    """Support for each item feature given probe and an instance-based memory."""

    t = trace_activation(memory_state, input_feature_pattern, trace_scale)
    a = jnp.dot(t, memory_state)[input_feature_pattern.shape[0] :]
    return power_scale(a, feature_scale)


@jit
@dispatch
def probe(
    memory: InstanceMcf, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Probe the memory with a probe vector"""
    return instance_probe(
        memory.state, input_feature_pattern, memory.feature_scale, memory.trace_scale
    )
