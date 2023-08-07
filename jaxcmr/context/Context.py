from flax.struct import PyTreeNode
from jaxtyping import Float, Array
from plum import dispatch
from jax import jit


class Context(PyTreeNode):
    pass
    

@jit
@dispatch.abstract
def start_context_input(context: Context) -> Float[Array, "context_feature_units"]:
    "Vector representation of start-of-list context."


@jit
@dispatch.abstract
def delay_context_input(context: Context) -> Float[Array, "context_feature_units"]:
    "Vector representation of out-of-list context."


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
def integrate_start_context(
    context: Context,
    drift_rate: float | Float[Array, ""],
) -> Context:
    "Integrate start-of-list context into current state of a temporal context representation"
    return integrate(context, start_context_input(context), drift_rate)


@jit
@dispatch
def integrate_delay_context(
    context: Context,
    drift_rate: float | Float[Array, ""],
) -> Context:
    "Integrate out-of-list context into current state of a temporal context representation"
    return integrate(context, delay_context_input(context), drift_rate)


