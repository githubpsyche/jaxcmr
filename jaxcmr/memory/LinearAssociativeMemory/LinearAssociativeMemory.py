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

# %% Imports

from jax import jit, numpy as jnp
from plum import dispatch

from jaxcmr.helpers import Float, Array, ScalarFloat, ScalarInteger, input_features, output_features
from jaxcmr.helpers import replace
from jaxcmr.memory import OneWayMemory, scale_activation

# %% Public interface

__all__ = [
    "LinearAssociativeMemory",
    "hebbian_associate",
    "associate",
    "probe",
]

# %% Types


class LinearAssociativeMemory(OneWayMemory, mutable=True):
    @dispatch
    def __init__(self, state: Float[Array, "input_features output_features"]):
        self.state = state

    @property
    def input_features(self) -> ScalarInteger:
        return self.state.shape[0]

    @property
    def output_features(self) -> ScalarInteger:
        return self.state.shape[1]


# %% Encoding


@jit
@dispatch
def hebbian_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: ScalarFloat,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "input_features output_features"]:
    """Associate input and output feature patterns in a M x N linear associative memory state"""
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern)
    )


@jit
@dispatch
def associate(
    memory: LinearAssociativeMemory,
    learning_rate: ScalarFloat,
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> LinearAssociativeMemory:
    """Associate input and output feature patterns in a linear associative memory"""
    return replace(
        memory,
        state=hebbian_associate(
            memory.state, learning_rate, input_feature_pattern, output_feature_pattern
        ),
    )


# %% Associative Recall


@jit
@dispatch
def linear_probe(
    memory_state: Float[Array, "input_features output_features"],
    _probe: Float[Array, "input_features"],
    scale: ScalarFloat = 1.0,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of a M x N linear associative memory state"""
    return scale_activation(jnp.dot(_probe, memory_state), scale)


@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory,
    _probe: Float[Array, "input_features"],
    scale: ScalarFloat = 1.0,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of a linear associative memory"""
    return linear_probe(memory.state, _probe, scale)
