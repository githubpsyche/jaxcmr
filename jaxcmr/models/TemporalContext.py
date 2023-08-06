"""
Temporal Context as specifed in the Context Maintenance and Retrieval model (CMR).
"""

from plum import dispatch
from flax.struct import PyTreeNode
from jaxtyping import Integer, Float, Array
import jax.numpy as jnp


class TemporalContext(PyTreeNode):
    state: Float[Array, "context_feature_units"]
    start_context_input: Float[Array, "context_feature_units"] 
    delay_context_input: Float[Array, "context_feature_units"]

@dispatch
def initialize_temporal_context(
    item_count: int | Integer[Array, ""]
) -> TemporalContext:
    "Initialize a temporal context representation."
    context_state = jnp.zeros(item_count + 2)
    return TemporalContext(
        state=context_state.at[0].set(1),
        start_context_input=context_state.at[0].set(1),
        delay_context_input=context_state.at[-1].set(1),
    )


@dispatch
def integrate(
    context_state: Float[Array, "context_feature_units"],
    context_input: Float[Array, "context_feature_units"],
    drift_rate: float | Float[Array, ""],
) -> Float[Array, "context_feature_units"]:
    "Integrate an input representation into a contextual state."
    rho = jnp.sqrt(
        1 + jnp.square(drift_rate) * (jnp.square(context_state * context_input) - 1)
    ) - (drift_rate * (context_state * context_input))

    return (rho * context_state) + (drift_rate * context_input)


@dispatch
def integrate(
    context: TemporalContext,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: float | Float[Array, ""],
) -> TemporalContext:
    "Integrate an input representation into current state of a temporal context representation."
    return context.replace(
        state=integrate(context.state, context_input, drift_rate))


@dispatch
def integrate_start_context(
    context: TemporalContext,
    drift_rate: float | Float[Array, ""],
) -> TemporalContext:
    "Integrate start-of-list context into current state of a temporal context representation."
    return integrate(context, context.start_context_input, drift_rate)


@dispatch
def integrate_delay_context(
    context: TemporalContext,
    drift_rate: float | Float[Array, ""],
) -> TemporalContext:
    "Integrate out-of-list context into current state of a temporal context representation."
    return integrate(context, context.delay_context_input, drift_rate)