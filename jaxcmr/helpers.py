from copy import copy
import jaxtyping
import numpy as np
from jax import numpy as jnp, jit, lax, config
from plum import dispatch
import numpy as np

config.update("jax_enable_x64", True)

# lb = jnp.finfo(jnp.float32).eps
lb = jnp.array(np.finfo(float).eps)

ScalarInteger = jaxtyping.Integer[jaxtyping.Array, ""] | int | np.int32 | np.int64
ScalarFloat = jaxtyping.Float[jaxtyping.Array, ""] | float | np.float32 | np.float64
ScalarBool = jaxtyping.Bool[jaxtyping.Array, ""] | bool | np.bool_
PRNGKeyArray = jaxtyping.PRNGKeyArray

study_events = "The number of (possible) study events in the simulated trial"
recall_outcomes = "The number of (possible) retrieval outcomes in the simulated trial"
context_feature_units = "Number of units in the context representation"
input_features = "Number of units in the input representation"
output_features = "Number of units in the output representation"
instances = "Number of instances in the memory"
recall_events = "Number of recall events in the simulated trial"

Integer = jaxtyping.Integer
Float = jaxtyping.Float
Array = jaxtyping.Array
Bool = jaxtyping.Bool


def replace(instance, **kwargs):
    new_instance = copy(instance)

    for attr, value in kwargs.items():
        setattr(new_instance, attr, value)

    return new_instance


@jit
@dispatch
def normalize(vector: Float[Array, "features"]) -> Float[Array, "output_features"]:
    """Enforce magnitude of vector to 1."""
    return vector / jnp.sqrt(jnp.sum(jnp.square(vector)) + lb)


@jit
@dispatch
def power_scale(
    vector: Float[Array, "output_features"], scale: ScalarFloat
) -> Float[Array, "output_features"]:
    """Scale activation vector by exponent factor using the logsumexp trick to avoid underflow."""
    log_activation = jnp.log(vector)
    return lax.cond(
        jnp.logical_and(jnp.any(vector != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: vector,
        None,
    )
