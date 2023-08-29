"""
Instance Mcf
"""

# %% Imports

from jax import jit, numpy as jnp
from functools import partial
from plum import dispatch
from jaxcmr.helpers import (
    ScalarFloat,
    ScalarInteger,
    Float,
    Array,
    input_features,
    output_features,
)
from jaxcmr.memory.types.OneWayMemory import InstanceMemory

# %% Exports

__all__ = [
    "InstanceMcf",
]

# %% Subtype of InstanceMemory


class InstanceMcf(InstanceMemory, mutable=True):
    @dispatch
    def __init__(
        self,
        state: Float[Array, "instances instance_features"],
        encoding_index: ScalarInteger = 0,
        feature_scale: ScalarFloat = 1.0,
        trace_scale: ScalarFloat = 1.0,
        _input_features: ScalarInteger = 0,
        _output_features: ScalarInteger = 0,
    ):
        self.state = state
        self.encoding_index = encoding_index
        self.feature_scale = feature_scale
        self.trace_scale = trace_scale
        self.input_features = _input_features
        self.output_features = _output_features

    @classmethod
    @dispatch
    def create(
        cls,
        item_count: ScalarInteger,
        presentation_count: ScalarInteger,
        shared_support: ScalarFloat,
        item_support: ScalarFloat,
        feature_scale: ScalarFloat = 1.0,
        trace_scale: ScalarFloat = 1.0,
    ):
        return cls(
            init_instance_mcf(
                item_count, presentation_count, shared_support, item_support
            ),
            item_count,
            feature_scale,
            trace_scale,
            item_count + 2,
            item_count,
        )


@partial(jit, static_argnums=(0, 1))
def init_instance_mcf(
    item_count: ScalarInteger,
    presentation_count: ScalarInteger,
    shared_support: ScalarFloat,
    item_support: ScalarFloat,
) -> Float[Array, "instances instance_features"]:
    """Initialize a instance-based context-to-feature memory state"""

    item_feature_count = item_count
    context_feature_count = item_count + 2

    items = (
        (jnp.eye(item_count) * item_support)
        - (jnp.eye(item_count, item_feature_count) * shared_support)
        + shared_support
    )
    contexts = jnp.eye(item_count, context_feature_count, 1)
    presentations = jnp.zeros(
        (presentation_count, context_feature_count + item_feature_count)
    )

    return jnp.vstack((jnp.hstack((contexts, items)), presentations))
