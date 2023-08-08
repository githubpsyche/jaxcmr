"""
Context
"""

#%% Imports

from flax.struct import PyTreeNode
from jaxtyping import Float, Array
from plum import dispatch
from jax import jit

#%% Public interface

__all__ = [
    "Context",
    "get_state",
    "set_state",
    "get_start_context_input",
    "get_delay_context_input",
    "integrate",
    "integrate_start_context",
    "integrate_delay_context",
]

#%% Type

class Context(PyTreeNode):
    pass
    
#%% Accessors

@jit
@dispatch.abstract
def set_state(context: Context, new_state: Float[Array, "context_feature_units"]) -> Context:
    "Update the state of a context representation"

@jit
@dispatch.abstract
def get_state(context: Context) -> Float[Array, "context_feature_units"]:
    "Return the state of a context representation as a 1D array"

@jit
@dispatch.abstract
def get_start_context_input(context: Context) -> Float[Array, "context_feature_units"]:
    "Vector representation of start-of-list context."


@jit
@dispatch.abstract
def get_delay_context_input(context: Context) -> Float[Array, "context_feature_units"]:
    "Vector representation of out-of-list context."

#%% Abstract methods

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
    return integrate(context, get_start_context_input(context), drift_rate)

@jit
@dispatch
def integrate_delay_context(context: Context, drift_rate: float | Float[Array, ""]) -> Context:
    "Integrate out-of-list context into current state of a temporal context representation"
    return integrate(context, get_delay_context_input(context), drift_rate)