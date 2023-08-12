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

from jaxtyping import Float, Array
from plum import dispatch
from jax import jit, lax, numpy as jnp
from jaxcmr.memory import OneWayMemory
from jaxcmr.helpers import replace

# %% Public interface

__all__ = [
    "LinearAssociativeMemory",
    "hebbian_associate",
    "associate",
    "probe",
    "scale_activation",
]

# %% Types


class LinearAssociativeMemory(OneWayMemory, mutable=True):
    @dispatch
    def __init__(self, state: Float[Array, "input_features output_features"]):
        self.state = state

    @property
    def input_features(self):
        return self.state.shape[0]

    @property
    def output_features(self):
        return self.state.shape[1]


# %% Encoding


@jit
@dispatch
def hebbian_associate(
    memory_state: Float[Array, "input_features output_features"],
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> Float[Array, "input_features output_features"]:
    "Associate input and output feature patterns in a M x N linear associative memory state"
    return memory_state + (
        learning_rate * jnp.outer(input_feature_pattern, output_feature_pattern)
    )


@jit
@dispatch
def associate(
    memory: LinearAssociativeMemory,
    learning_rate: float | Float[Array, ""],
    input_feature_pattern: Float[Array, "input_features"],
    output_feature_pattern: Float[Array, "output_features"],
) -> LinearAssociativeMemory:
    "Associate input and output feature patterns in a linear associative memory"
    return replace(
        memory,
        state=hebbian_associate(
            memory.state, learning_rate, input_feature_pattern, output_feature_pattern
        ),
    )


# %% Associative Recall


@jit
@dispatch
def scale_activation(
    activation: Float[Array, "output_features"], scale: float | Float[Array, ""]
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
def linear_probe(
    memory_state: Float[Array, "input_features output_features"],
    probe: Float[Array, "input_features"],
) -> Float[Array, "output_features"]:
    "Return the activation vector of a M x N linear associative memory state"
    return jnp.dot(probe, memory_state)

@jit
@dispatch
def linear_probe(
    memory_state: Float[Array, "input_features output_features"],
    probe: Float[Array, "input_features"],
    scale: float | Float[Array, ""],
) -> Float[Array, "output_features"]:
    "Return the scaled activation vector of a M x N linear associative memory state"
    return scale_activation(jnp.dot(probe, memory_state), scale)

@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory, probe: Float[Array, "input_features"]
) -> Float[Array, "output_features"]:
    "Return the activation vector of a linear associative memory"
    return linear_probe(memory.state, probe)


@jit
@dispatch
def probe(
    memory: LinearAssociativeMemory,
    probe: Float[Array, "input_features"],
    scale: float | Float[Array, ""],
) -> Float[Array, "output_features"]:
    "Return the scaled activation vector of a linear associative memory"
    return linear_probe(memory.state, probe, scale)
