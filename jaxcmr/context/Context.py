"""
Context
"""

from simple_pytree import Pytree
from jaxtyping import Float, Array
from plum import dispatch
from jax import jit

__all__ = [
    "Context",
    "integrate",
    "integrate_start_context",
    "integrate_delay_context",
]

class Context(Pytree):
    start_context_input: Float[Array, "context_feature_units"]
    delay_context_input: Float[Array, "context_feature_units"]

@jit
@dispatch.abstract
def integrate(
    context: Context,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: float | Float[Array, ""],
) -> Context:
    "Integrate an input representation into current state of a context representation"

@jit
@dispatch
def integrate_start_context(context: Context, drift_rate: float | Float[Array, ""]) -> Context:
    "Integrate start-of-list context into current state of a temporal context representation"
    return integrate(context, context.start_context_input, drift_rate)

@jit
@dispatch
def integrate_delay_context(context: Context, drift_rate: float | Float[Array, ""]) -> Context:
    "Integrate out-of-list context into current state of a temporal context representation"
    return integrate(context, context.delay_context_input, drift_rate)