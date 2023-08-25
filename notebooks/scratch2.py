# %%

from functools import partial
from jax import jit, numpy as jnp
import numpy as np


def trial_item_count(presentation):
    "Return the number of unique items in a presentation sequence"
    return np.max(presentation)


def zeros(item_count):
    "Return a vector of zeros with the specified size"
    return jnp.zeros(item_count)


@jit
def zeros_like_item_count(presentation):
    "Return a vector of zeros with length equal to the number of unique items in a trial"
    print(presentation)
    return zeros(trial_item_count(presentation))


presentation = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16,
                          17, 10, 18, 19, 20, 19, 21, 22, 23, 20, 24, 25, 26, 22, 27, 28, 24,
                          29, 30, 31, 32, 33, 34])
zeros_like_item_count(presentation)

# %%
