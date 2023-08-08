"""
OneWayMemory
Abstract type and functions for one-way associative memories.
Subclasses must implement the following functions:
    input_features
    output_features
    associate
    probe
"""

from jaxtyping import Float, Array
from flax.struct import PyTreeNode
from plum import dispatch

class OneWayMemory(PyTreeNode):
    pass

@dispatch.abstract
def input_features(memory: OneWayMemory) -> int:
    "Return the number of input features of a one-way associative memory"

@dispatch.abstract
def output_features(memory: OneWayMemory) -> int:
    "Return the number of output features of a one-way associative memory"

@dispatch.abstract
def associate(
    memory: OneWayMemory, 
    input_feature_pattern: Float[Array, "input_features"], 
    output_feature_pattern: Float[Array, "output_features"]
    ) -> Float[Array, "output_features"]:
    "Associate input and output feature patterns in a one-way associative memory"

@dispatch.abstract
def probe(memory: OneWayMemory, input: Float[Array, "input_features"]
    ) -> Float[Array, "output_features"]:
    "Retrieve the output vector associated with an input vector in a one-way associative memory"