"""
OneWayMemory
Abstract type and functions for one-way associative memories.
"""

from jaxcmr.helpers import Float, Integer, Array, ScalarInteger, ScalarFloat
from jax import jit, lax, numpy as jnp
from simple_pytree import Pytree
from plum import dispatch

__all__ = ["OneWayMemory", "associate", "probe", "scale_activation"]


class OneWayMemory(Pytree, mutable=True):
    input_features: ScalarInteger
    output_features: ScalarInteger


@dispatch.abstract
def associate(
    memory: OneWayMemory,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "output_features"]:
    "Associate input and output feature patterns in a one-way associative memory"


@dispatch.abstract
def probe(
    memory: OneWayMemory, input: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    "Retrieve the output vector associated with an input vector in a one-way associative memory"

@jit
@dispatch
def scale_activation(
    activation: Float[Array, "output_features"], 
    scale: ScalarFloat
) -> Float[Array, "output_features"]:
    "Scale activation vector by a exponent factor using the logsumexp trick to avoid underflow."
    log_activation = jnp.log(activation)
    return lax.cond(
        jnp.logical_and(jnp.any(activation != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: activation,
        None,
    )