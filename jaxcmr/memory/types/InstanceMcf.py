"""
Instance Mcf
"""

# %% Imports

from functools import partial

from jax import jit, numpy as jnp
from plum import dispatch

from jaxcmr.helpers import (
    ScalarFloat,
    ScalarInteger,
    Float,
    Array,
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
        encoding_index: ScalarInteger,
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

    @classmethod
    @dispatch
    def create(
        cls,
        items,
        presentation_count: ScalarInteger,
        parameters: dict,
    ):
        return cls.create(
            items,
            presentation_count,
            parameters["shared_support"],
            parameters["item_support"],
            parameters["choice_sensitivity"],
            parameters["mcf_trace_sensitivity"],
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
