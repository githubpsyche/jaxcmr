"""
Context

A type hierarchy for implementing contextual representations that mediate encoding and memory search during tasks
such as free recall.

Context can `integrate` novel contextual features to change its state over time.

Within retrieved context accounts of performance, context can mediate activities such as learning and memory search.
For example, its state at each study position can be associated in memory with active item features. And at each
recall attempt, its state can be used to probe memory, directing recall towards item that were studied with similar
contextual states.

These constraints call for a context class that can integrate new features and queried for its current state based on
parameters configured at the time of its instantiation. When an operation depends on information that would not be
available at the time of instantiation, this information should be passed as an argument to the operation.

To support drift back toward its initial state or away from study list context, a context class should also support
the ability to integrate a start-of-list context and an out-of-list context."""

# %% Setup

from simple_pytree import Pytree
from jaxcmr.helpers import Float, Array, ScalarFloat, context_feature_units
from plum import dispatch
from jax import jit

__all__ = [
    "Context",
    "integrate",
    "integrate_start_context",
    "integrate_outlist_context",
]


# %% Type and expected operations


class Context(Pytree):
    start_context_input: Float[Array, "context_feature_units"]
    outlist_context_input: Float[Array, "context_feature_units"]
    state: Float[Array, "context_feature_units"]


# %% Abstract Operations


@jit
@dispatch.abstract
def integrate(
    context: Context,
    context_input: Float[Array, "context_feature_units"],
    drift_rate: ScalarFloat,
) -> Context:
    """Integrate an input representation into current state of a context representation"""


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
