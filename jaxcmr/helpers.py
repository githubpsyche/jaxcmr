from copy import copy
import jaxtyping
import numpy as np
from jax import numpy as jnp

lb = jnp.finfo(jnp.float32).eps

ScalarInteger = jaxtyping.Integer[jaxtyping.Array, ''] | int | np.int32 | np.int64
ScalarFloat = jaxtyping.Float[jaxtyping.Array, ''] | float | np.float32 | np.float64
ScalarBool = jaxtyping.Bool[jaxtyping.Array, ''] | bool | np.bool_
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
