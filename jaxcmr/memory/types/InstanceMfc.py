"""
Instance Mfc

Not necessary per se for implementing InstanceCMR, but useful for exploring ideas about how different experiences separately influence model predictions.
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
    "InstanceMfc",
]

# %% Subtype of InstanceMemory


class InstanceMfc(InstanceMemory, mutable=True):

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
        learning_rate: ScalarFloat,
        feature_scale: ScalarFloat = 1.0,
        trace_scale: ScalarFloat = 1.0,
    ):
        return cls(
            init_instance_mfc(item_count, presentation_count, learning_rate),
            item_count,
            feature_scale,
            trace_scale,
            item_count,
            item_count + 2,
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
            parameters['learning_rate'],
            parameters['mfc_choice_sensitivity'],
            parameters['mfc_trace_sensitivity']
        )
    

@partial(jit, static_argnums=(0, 1))
def init_instance_mfc(
    item_count: ScalarInteger, presentation_count: ScalarInteger, learning_rate: ScalarFloat
) -> Float[Array, "instances instance_features"]:
    """An instance-based feature-to-context memory assuming one-hot item representations."""

    item_feature_count = item_count
    context_feature_count = item_count + 2

    items = jnp.eye(item_count)
    contexts = jnp.eye(item_count, context_feature_count, 1) * learning_rate
    presentations = jnp.zeros((presentation_count, item_feature_count + context_feature_count))
    
    return jnp.vstack(jnp.hstack((items, contexts)), presentations)


