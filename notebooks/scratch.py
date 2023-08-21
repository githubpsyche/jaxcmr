#%%

from jaxcmr.evaluation import extract_objective_data
from jaxcmr.datasets import load_data, generate_trial_mask
import numpy as np
from jax import jit, random, lax, numpy as jnp

data_path = '../data/{}.h5'
data = load_data(data_path.format('LohnasKahana2014'))

trial_query = "data['list_type'] != -1"
trial_mask = generate_trial_mask(data, trial_query)
trials, list_lengths, presentations, pres_string_ids, has_repetitions = extract_objective_data(data, trial_mask)

presentation = presentations[0][0]

#%%
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

presentation = jnp.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 12, 13, 14, 15, 16,
       17, 10, 18, 19, 20, 19, 21, 22, 23, 20, 24, 25, 26, 22, 27, 28, 24,
       29, 30, 31, 32, 33, 34])
zeros_like_item_count(presentation)

# %%

trial_item_count(presentation)

# %%



#%%


@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
# %%
