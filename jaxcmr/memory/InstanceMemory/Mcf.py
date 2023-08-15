"""
Instance-based $M^{CF}$

Initialization functions for a context-to-feature instance-based memory that reproduces the behavior of the linear associative $M^{CF}$ memory as specified for CMR.
"""

# %% Imports

from jaxtyping import Integer, Float, Array
from plum import dispatch
from jax import jit, lax, numpy as jnp
from functools import partial
from jaxcmr.memory.InstanceMemory.InstanceMemory import (
    InstanceMemory,
    instance_probe,
)

# %% Public interface

__all__ = [
    "InstanceMcf",
    "probe",
]

# %% Subtype of InstanceMemory

class InstanceMcf(InstanceMemory, mutable=True):

    @dispatch
    def __init__(
        self, 
        state: Float[Array, "instances instance_features"], 
        encoding_index: int | Integer[Array, ""] = 0,
        feature_scale: float | Float[Array, ""] = 1.0,
        trace_scale: float | Float[Array, ""] = 1.0,
        input_features: int | Integer[Array, ""] = 0,
        output_features: int | Integer[Array, ""] = 0,
        ):
        self.state = state
        self.encoding_index = encoding_index
        self.feature_scale = feature_scale
        self.trace_scale = trace_scale
        self.input_features = input_features
        self.output_features = output_features

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: int | Integer[Array, ""],
        presentation_count: int | Integer[Array, ""],
        shared_support: float | Float[Array, ""],
        item_support: float | Float[Array, ""],
        feature_scale: float | Float[Array, ""] = 1.0,
        trace_scale: float | Float[Array, ""] = 1.0,
    ):
        return cls(
            basic_init_instance_mcf(item_count, presentation_count, shared_support, item_support),
            item_count,
            feature_scale,
            trace_scale,
            item_count +2, 
            item_count,
        )
    
@partial(jit, static_argnums=(0,1))
def basic_init_instance_mcf(
    item_count: int | Integer[Array, ""],
    presentation_count: int | Integer[Array, ""],
    shared_support: float | Float[Array, ""],
    item_support: float | Float[Array, ""],
) -> Float[Array, "context_features item_features"]:
    "Initialize a instance-based context-to-feature memory state"

    item_feature_count = item_count
    context_feature_count = item_count + 2

    items = (jnp.eye(item_count) * item_support) - (
        jnp.eye(item_count, item_feature_count) * shared_support) + shared_support
    contexts = jnp.eye(item_count, context_feature_count, 1)
    presentations = jnp.zeros((presentation_count, context_feature_count + item_feature_count))

    return jnp.vstack((jnp.hstack((contexts, items)), presentations))

# %% Probe


@jit
@dispatch
def probe(
    memory: InstanceMcf,
    probe: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    "Probe the memory with a probe vector"
    return instance_probe(
        memory.state, probe, memory.feature_scale, memory.trace_scale
        )