"""
OneWayMemory
Abstract type and functions for one-way associative memories.
"""

from jax import jit, lax, numpy as jnp
from plum import dispatch
from simple_pytree import Pytree

from jaxcmr.helpers import (
    Float, Array, ScalarInteger, ScalarFloat, input_features, output_features, lb)

__all__ = ["OneWayMemory", "associate", "probe"]


class OneWayMemory(Pytree, mutable=True):
    input_features: ScalarInteger
    output_features: ScalarInteger


@dispatch.abstract
def associate(
    memory: OneWayMemory,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "output_features"]:
    """Associate input and output feature patterns in a one-way associative memory"""


@dispatch.abstract
def probe(
    memory: OneWayMemory, input_feature_pattern: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    """Retrieve the output vector associated with an input vector in a one-way associative memory"""


