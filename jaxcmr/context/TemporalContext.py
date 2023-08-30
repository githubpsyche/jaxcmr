"""
Temporal Context

Temporal context as specified in the Context Maintenance and Retrieval (CMR) model.

Temporal context initializes with a state vector of zeros with a 1 in the first position (TemporalContext.create).
This also initializes start_context_input same state vector.
Successive positions reserve a unit to represent each item in the study list.
An final contextual unit is reserved to represent out-of-list context (outlist_context_input).

This state vector is updated by integrating novel context features such that the resulting state vector is a weighted
sum of the current state vector and the novel context features, maintaining a unit magnitude (see `rho_integrate`)."""

from plum import dispatch
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarFloat,
    ScalarInteger,
    context_feature_units,
    replace,
)
from jax import jit, numpy as jnp
from jaxcmr.context.Context import (
    Context,
    integrate_start_context,
    integrate_outlist_context,
)

__all__ = [
    "TemporalContext",
    "integrate",
    "rho_integrate",
]


class TemporalContext(Context, mutable=True):
    def __init__(
        self,
        state: Float[Array, "context_feature_units"],
        start_context_input: Float[Array, "context_feature_units"],
        outlist_context_input: Float[Array, "context_feature_units"],
    ):
        self.state = state
        self.start_context_input = start_context_input
        self.outlist_context_input = outlist_context_input

    @classmethod
    @dispatch
    def create(cls, item_count: ScalarInteger):
        context_state = jnp.zeros(item_count + 2)
        return cls(
            state=context_state.at[0].set(1),
            start_context_input=context_state.at[0].set(1),
            outlist_context_input=context_state.at[-1].set(1),
        )


@jit
@dispatch
def rho_integrate(
    context_state: Float[Array, "context_feature_units"],
    context_input: Float[Array, "context_feature_units"],
    drift_rate: ScalarFloat,
) -> Float[Array, "context_feature_units"]:
    """Apply rho integration rule to update context state"""
    rho = jnp.sqrt(
        1 + jnp.square(drift_rate) * (jnp.square(context_state * context_input) - 1)
    ) - (drift_rate * (context_state * context_input))
    return (rho * context_state) + (drift_rate * context_input)


@jit
@dispatch
def integrate(
    context: TemporalContext,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: ScalarFloat,
) -> TemporalContext:
    """Integrate an input representation into temporal context"""
    return replace(
        context, state=rho_integrate(context.state, context_input, drift_rate)
    )
