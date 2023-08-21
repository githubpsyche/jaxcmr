from copy import copy
import jaxtyping
import numpy as np

ScalarInteger = jaxtyping.Integer[jaxtyping.Array, ''] | int | np.int32 | np.int64
ScalarFloat = jaxtyping.Float[jaxtyping.Array, ''] | float | np.float32 | np.float64
ScalarBool = jaxtyping.Bool[jaxtyping.Array, ''] | bool | np.bool_
PRNGKeyArray = jaxtyping.PRNGKeyArray

Integer = jaxtyping.Integer
Float = jaxtyping.Float
Array = jaxtyping.Array
Bool = jaxtyping.Bool

def replace(instance, **kwargs):
    new_instance = copy(instance)

    for attr, value in kwargs.items():
        setattr(new_instance, attr, value)

    return new_instance

