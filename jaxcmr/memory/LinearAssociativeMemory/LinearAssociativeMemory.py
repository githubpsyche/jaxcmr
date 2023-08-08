"""
Linear Associative Memory
"""

#%% Imports

from jaxtyping import Float, Array
from plum import dispatch
from jax import jit, lax, numpy as jnp
from jaxcmr.memory import OneWayMemory

#%% Public interface

__all__ = [
    'LinearAssociativeMemory',
    'state',
    'input_features',
    'output_features',
    'update_state',
    'hebbian_associate',
    'associate',
    'probe',
    'scale_activation',
    ]

#%% Types

class LinearAssociativeMemory(OneWayMemory):
    state: Float[Array, "input_features output_features"]

#%% Accessors

@jit
@dispatch
def state(memory: LinearAssociativeMemory) -> Float[Array, "input_features output_features"]:
    "Return the state of a linear associative memory as a 2D array"
    return memory.state

@jit
@dispatch
def input_features(memory: LinearAssociativeMemory) -> int:
    "Return the number of input features of a linear associative memory"
    return state(memory).shape[0]

@jit
@dispatch
def output_features(memory: LinearAssociativeMemory) -> int:
    "Return the number of output features of a linear associative memory"
    return state(memory).shape[1]

#%% Encoding

@jit
@dispatch
def update_state(
    memory: LinearAssociativeMemory, 
    new_state: Float[Array, "input_features output_features"]
    ) -> LinearAssociativeMemory:
    "Update the state of a linear associative memory"
    return LinearAssociativeMemory(new_state)


@jit
@dispatch
def hebbian_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"]
    ) -> Float[Array, "input_features output_features"]:
    "Associate input and output feature patterns in a M x N linear associative memory state"
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern
        ))


@jit
@dispatch
def associate(
    memory: LinearAssociativeMemory,
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"]
    ) -> LinearAssociativeMemory:
    "Associate input and output feature patterns in a linear associative memory"
    return update_state(
        memory, 
        hebbian_associate(
            state(memory), learning_rate, input_feature_pattern, output_feature_pattern)
        )

#%% Associative Recall

@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory, 
    probe: Float[Array, "input_features"]
    ) -> Float[Array, "output_features"]:
    "Return the activation vector of a linear associative memory"
    return jnp.dot(probe, state(memory))

@jit
@dispatch
def scale_activation(
    activation: Float[Array, "output_features"],
    scale: float | Float[Array, ""]
    ) -> Float[Array, "output_features"]:
    "Scale activation vector by a exponent factor using the logsumexp trick to avoid underflow."
    log_activation = jnp.log(activation)

    return lax.cond(
        jnp.logical_and(jnp.any(activation != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: activation,
        None,
    )

@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory,
    probe: Float[Array, "input_features"],
    scale: float | Float[Array, ""]
    ) -> Float[Array, "output_features"]:
    "Return the scaled activation vector of a linear associative memory"
    return scale_activation(probe(memory, probe), scale)
