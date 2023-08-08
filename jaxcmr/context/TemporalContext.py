from plum import dispatch
from jaxtyping import Integer, Float, Array
from jax import jit, numpy as jnp
from jaxcmr.context.Context import Context
from functools import partial

#%% Type

class TemporalContext(Context):
    state: Float[Array, "context_feature_units"]
    start_context_input: Float[Array, "context_feature_units"] 
    delay_context_input: Float[Array, "context_feature_units"]

#%% Public interface

__all__ = [
    "TemporalContext",
    "state",
    "start_context_input",
    "delay_context_input",
    "integrate",
    "rho_integrate",
    "initialize_temporal_context",
]

#%% Accessors

@jit
@dispatch
def state(context: TemporalContext) -> Float[Array, "context_feature_units"]:
    return context.state


@jit
@dispatch
def start_context_input(context: TemporalContext) -> Float[Array, "context_feature_units"]:
    return context.start_context_input


@jit
@dispatch
def delay_context_input(context: TemporalContext) -> Float[Array, "context_feature_units"]:
    return context.delay_context_input


@jit
@dispatch
def update_state(
    context: TemporalContext, new_state: Float[Array, "context_feature_units"],
) -> TemporalContext:
    return context.replace(state=new_state)


#%% Specialized functions

@partial(jit, static_argnums=(0,))
@dispatch
def initialize_temporal_context(item_count: int | Integer[Array, ""]) -> TemporalContext:
    context_state = jnp.zeros(item_count + 2)
    return TemporalContext(
        state=context_state.at[0].set(1),
        start_context_input=context_state.at[0].set(1),
        delay_context_input=context_state.at[-1].set(1),
    )


@jit
@dispatch
def rho_integrate(
    context_state: Float[Array, "context_feature_units"],
    context_input: Float[Array, "context_feature_units"],
    drift_rate: float | Float[Array, ""],
) -> Float[Array, "context_feature_units"]:
    "Apply rho integration rule to update context state"
    rho = jnp.sqrt(
        1 + jnp.square(drift_rate) * (jnp.square(context_state * context_input) - 1)
    ) - (drift_rate * (context_state * context_input))
    return (rho * context_state) + (drift_rate * context_input)


@jit
@dispatch
def integrate(
    context: TemporalContext,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: float | Float[Array, ""],
) -> TemporalContext:
    "Integrate an input representation into temporal context"
    return update_state(context, rho_integrate(state(context), context_input, drift_rate))
