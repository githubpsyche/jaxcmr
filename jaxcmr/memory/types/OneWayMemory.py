"""
OneWayMemory
Abstract type and functions for one-way associative memories.
"""

# %% Imports

from plum import dispatch
from simple_pytree import Pytree

from jaxcmr.helpers import (
    Float,
    Array,
    ScalarInteger,
)

# %% Exports

__all__ = ["OneWayMemory", "LinearAssociativeMemory", "InstanceMemory"]

# %% Base Type Hierarchy for One-Way Associative Memories


class OneWayMemory(Pytree, mutable=True):
    """Abstract type with declared attributes for one-way associative memories"""

    input_features: ScalarInteger
    output_features: ScalarInteger


class LinearAssociativeMemory(OneWayMemory, mutable=True):
    """
    Concrete type and functions for one-way linear associative memory models.

    Input and output feature patterns are 1D arrays of shape (M,) and (N,),
    respectively.
    State interface provides a 2D array of shape (M, N).
    """

    @dispatch
    def __init__(self, state: Float[Array, "input_features output_features"]):
        self.state = state

    @property
    def input_features(self) -> ScalarInteger:
        return self.state.shape[0]

    @property
    def output_features(self) -> ScalarInteger:
        return self.state.shape[1]


class InstanceMemory(OneWayMemory, mutable=True):
    """
    Concrete type and functions for one-way instance-based memory models based on MINERVA 2.

    These store and retrieve specific memory instances instead of learning association weights.
    Context and item patterns are associated by concatenating them into a single vector
    and storing them in a stack of discrete memory traces.

    To retrieve associations, a probe is compared to each trace in the stack
    based on their cosine similarity.
    From there, the similarity-weighted average of traces forms an overall activation vector.
    The memory returns just the portion of the activation vector corresponding to feature
    dimensions for the target pattern type.
    """

    @dispatch
    def __init__(
        self,
        state: Float[Array, "instances instance_features"],
        encoding_index: ScalarInteger = 0,
        _input_features: ScalarInteger = 0,
        _output_features: ScalarInteger = 0,
    ):
        self.state = state
        self.encoding_index = encoding_index
        self.input_features = _input_features
        self.output_features = _output_features
