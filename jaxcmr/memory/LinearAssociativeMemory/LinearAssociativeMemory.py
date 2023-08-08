"""
Linear Associative Memory
Concrete type and functions for one-way linear associative memory models.

Input and output feature patterns are 1D arrays of shape (M,) and (N,), 
respectively.
State interface provides a 2D array of shape (M, N).

Following principles of behavioral over implementational inheritance,
as long as subtypes provide a working implementation of the state and 
update_state functions, all other functions will work, regardless of subtype 
implementation.
"""

#%% Imports

from jaxtyping import Float, Array
from plum import dispatch
from jax import jit, lax, numpy as jnp
from jaxcmr.memory import OneWayMemory

#%% Public interface

__all__ = [
    'LinearAssociativeMemory',
    'get_state',
    'input_features',
    'output_features',
    'set_state',
    'hebbian_associate',
    'associate',
    'probe',
    'scale_activation',
    ]

#%% Types

class LinearAssociativeMemory(OneWayMemory):
    state: Float[Array, "input_features output_features"]

#%% Implementation-Coupled Getters and Setters

@jit
@dispatch
def get_state(memory: LinearAssociativeMemory) -> Float[Array, "input_features output_features"]:
    "Return the state of a linear associative memory as a 2D array"
    return memory.state

@jit
@dispatch
def set_state(
    memory: LinearAssociativeMemory, 
    new_state: Float[Array, "input_features output_features"]
    ) -> LinearAssociativeMemory:
    "Update the state of a linear associative memory"
    return LinearAssociativeMemory(new_state)

#%% Other Getters

@jit
@dispatch
def input_features(memory: LinearAssociativeMemory) -> int:
    "Return the number of input features of a linear associative memory"
    return get_state(memory).shape[0]

@jit
@dispatch
def output_features(memory: LinearAssociativeMemory) -> int:
    "Return the number of output features of a linear associative memory"
    return get_state(memory).shape[1]

#%% Encoding

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
    return set_state(
        memory, 
        hebbian_associate(
            get_state(memory), learning_rate, input_feature_pattern, output_feature_pattern)
        )

#%% Associative Recall

@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory, 
    probe: Float[Array, "input_features"]
    ) -> Float[Array, "output_features"]:
    "Return the activation vector of a linear associative memory"
    return jnp.dot(probe, get_state(memory))

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
