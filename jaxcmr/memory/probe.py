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
    InstanceMemory,
    LinearAssociativeMcf,
    InstanceMcf,
    LinearAssociativeMfc,
)
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarFloat,
    input_features,
    output_features,
    instances,
    power_scale,
    normalize,
)

# %% Exports

__all__ = ["probe", "linear_probe", "instance_probe", "trace_activation"]


# %% Abstract Probe Function


@dispatch.abstract
def probe(
    memory: OneWayMemory, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Retrieve the output vector associated with an input vector in a one-way associative memory"""


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
    scale: ScalarFloat,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of a M x N linear associative memory state"""
    return normalize(power_scale(jnp.dot(input_feature_pattern, memory_state), scale))


@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory,
    input_feature_pattern: Float[Array, "input_features"],
    scale: ScalarFloat,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of a linear associative memory"""
    return linear_probe(memory.state, input_feature_pattern, scale)


@jit
@dispatch
def probe(
    memory: LinearAssociativeMfc, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Probe a feature-to-context linear associative memory, normalizing the output."""
    return normalize(jnp.dot(input_feature_pattern, memory.state))


@jit
@dispatch
def probe(
    memory: LinearAssociativeMcf, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Probe a LinearAssociativeMcf, scaling output based using its sensitivity parameter."""
    return power_scale(
        jnp.dot(input_feature_pattern, memory.state), memory.choice_sensitivity
    )


# %% Probe Functions for Instance-Based Memories


@jit
@dispatch
def trace_activation(
    state: Float[Array, "instances instance_features"],
    input_feature_pattern: Float[Array, "input_features"],
    trace_scale: ScalarFloat,
) -> Float[Array, "instances"]:
    """Return activation for each trace in an instance-based memory"""
    a = jnp.dot(state[:, : input_feature_pattern.shape[0]], input_feature_pattern)
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


@jit
@dispatch
def probe(
    memory: InstanceMemory,
    input_feature_pattern: Float[Array, "input_features"],
    feature_scale: ScalarFloat = 1.0,
    trace_scale: ScalarFloat = 1.0,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of an instance-based memory"""
    return instance_probe(
        memory.state, input_feature_pattern, feature_scale, trace_scale
    )
