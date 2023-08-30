"""
Context

A type hierarchy for implementing contextual representations that mediate encoding and memory search during tasks such as free recall.

Context can `integrate` novel contextual features to change its state over time.

To support drift back toward its initial state or away from study list context, 
a context should also support integration of a start-of-list context and an out-of-list context.
"""

# %% Setup

from simple_pytree import Pytree
from jaxcmr.helpers import (
    Float,
    Array,
    ScalarFloat,
    ScalarInteger,
    context_feature_units,
    replace,
    normalize_to_unit_length
)
from plum import dispatch
from jax import jit, numpy as jnp

__all__ = [
    "Context",
    "TemporalContext",
    "integrate",
    "rho_integrate",
    "integrate_start_context",
    "integrate_outlist_context",
]


# %% Type and expected operations


class Context(Pytree):
    start_context_input: Float[Array, "context_feature_units"]
    outlist_context_input: Float[Array, "context_feature_units"]
    state: Float[Array, "context_feature_units"]

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

# %% Integration


@jit
@dispatch
def rho_integrate(
    context_state: Float[Array, "context_feature_units"],
    context_input: Float[Array, "context_feature_units"],
    drift_rate: ScalarFloat,
) -> Float[Array, "context_feature_units"]:
    """Integrate an input representation into a contextual state, preserving unit length."""
    context_input = normalize_to_unit_length(context_input)
    rho = jnp.sqrt(
        1 + jnp.square(drift_rate) * (jnp.square(context_state * context_input) - 1)
    ) - (drift_rate * (context_state * context_input))
    return (rho * context_state) + (drift_rate * context_input)


@jit
@dispatch
def integrate(
    context: Context,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: ScalarFloat,
) -> Context:
    """Integrate an input representation into current state of a context representation"""
    return replace(
        context, state=rho_integrate(context.state, context_input, drift_rate)
    )


@jit
@dispatch
def integrate_start_context(context: Context, drift_rate: ScalarFloat) -> Context:
    """Integrate start-of-list context into current state of a temporal context representation"""
    return integrate(context, context.start_context_input, drift_rate)


@jit
@dispatch
def integrate_outlist_context(context: Context, drift_rate: ScalarFloat) -> Context:
    """Integrate out-of-list context into current state of a temporal context representation"""
    return integrate(context, context.outlist_context_input, drift_rate)
