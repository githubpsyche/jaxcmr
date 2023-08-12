from plum import dispatch
from jaxtyping import Integer, Float, Array
from jax import jit, numpy as jnp
from jaxcmr.context.Context import Context, integrate_start_context, integrate_delay_context
from jaxcmr.helpers import replace

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
        delay_context_input: Float[Array, "context_feature_units"],
    ):
        self.state = state
        self.start_context_input = start_context_input
        self.delay_context_input = delay_context_input

    @classmethod
    @dispatch
    def create(cls, item_count: int | Integer[Array, ""]):
        context_state = jnp.zeros(item_count + 2)
        return cls(
            state=context_state.at[0].set(1), 
            start_context_input=context_state.at[0].set(1), 
            delay_context_input=context_state.at[-1].set(1)
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
    return replace(context, state=rho_integrate(context.state, context_input, drift_rate))
