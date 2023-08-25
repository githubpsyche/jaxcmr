"""
Instance Memory

Another popular class of memory models are instance-based memory models, which store and retrieve specific instances of
items and contexts instead of learning association weights.

Here we provide a simple implementation of an instance-based memory based on the MINERVA 2 model [@hintzman1984minerva,
@hintzman1986schema, @hintzman1988judgments] that associates patterns by concatenating them into a single vector and
storing them in a stack of discrete memory traces.

To retrieve associations, a probe is compared to each trace in the stack based on their cosine similarity. 
From there, the similarity-weighted average of traces forms an overall activation vector. 
The memory returns just the portion of the activation vector corresponding to feature dimensions for the target pattern
type.

"""

# %% Imports

from jaxcmr.helpers import Float, Array, ScalarInteger, ScalarFloat, input_features, output_features, instances
from plum import dispatch
from jax import jit, numpy as jnp
from jaxcmr.memory import OneWayMemory, scale_activation
from jaxcmr.helpers import replace

# %% Public Interface

__all__ = [
    "InstanceMemory",
    "instance_associate",
    "instance_probe",
    "trace_activation",
    "associate",
    "probe",
]


# %% Types


class InstanceMemory(OneWayMemory, mutable=True):
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


# %% Encoding


@jit
@dispatch
def associate(
        memory: InstanceMemory,
        learning_rate: ScalarFloat,
        input_pattern: Float[Array, "input_features"],
        output_pattern: Float[Array, "output_features"],
) -> InstanceMemory:
    """Associate input and output feature patterns in a instance-based memory"""
    return replace(
        memory,
        state=instance_associate(
            memory.state,
            memory.encoding_index,
            learning_rate,
            input_pattern,
            output_pattern,
        ),
        encoding_index=memory.encoding_index + 1,
    )


@jit
@dispatch
def instance_associate(
        memory_state: Float[Array, "instances instance_features"],
        encoding_index: ScalarInteger,
        learning_rate: ScalarFloat,
        input_pattern: Float[Array, "input_features"],
        output_pattern: Float[Array, "output_features"],
) -> Float[Array, "instances instance_features"]:
    """Associate two patterns in a instance-based memory state"""
    return memory_state.at[encoding_index].set(
        jnp.concatenate((input_pattern, output_pattern * learning_rate))
    )


# %% Associative Recall


@jit
@dispatch
def probe(
        memory: InstanceMemory,
        _probe: Float[Array, "input_features"],
        feature_scale: ScalarFloat = 1.0,
        trace_scale: ScalarFloat = 1.0,
) -> Float[Array, "output_features"]:
    """Return the scaled activation vector of an instance-based memory"""
    return instance_probe(memory.state, _probe, feature_scale, trace_scale)


@jit
@dispatch
def instance_probe(
        memory_state: Float[Array, "instances instance_features"],
        _probe: Float[Array, "input_features"],
        feature_scale: ScalarFloat,
        trace_scale: ScalarFloat,
) -> Float[Array, "output_features"]:
    """Support for each item feature given probe and an instance-based memory."""

    t = trace_activation(memory_state, _probe, trace_scale)
    a = jnp.dot(t, memory_state)[_probe.shape[0]:]
    return scale_activation(a, feature_scale)


@jit
@dispatch
def trace_activation(
        state: Float[Array, "instances instance_features"],
        _probe: Float[Array, "input_features"],
        trace_scale: ScalarFloat,
) -> Float[Array, "instances"]:
    """Return activation for each trace in an instance-based memory"""
    a = jnp.dot(state[:, : _probe.shape[0]], _probe)
    return scale_activation(a, trace_scale)
