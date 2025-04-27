# %%
# | default_exp simulation
#!%load_ext autoreload
#!%autoreload 2

# %% [markdown]
"""
# Model Simulation
"""

#%%

import json
from jaxcmr.helpers import load_data, generate_trial_mask
from jaxcmr.simulation import simulate_h5_from_h5, MemorySearchSimulator, preallocate_for_h5_dataset
from jaxcmr.models.cmr import BaseCMRFactory
from jax import random, jit
import jax.numpy as jnp

# %%

data_path = "data/HealeyKahana2014.h5"
fit_path = "fits/HealeyKahana2014_BaseCMR_best_of_3.json"
data_query = "data['listtype'] == -1"
seed = 0
rng = random.PRNGKey(seed)
data = load_data(data_path)
trial_mask = generate_trial_mask(data, data_query)
connections = jnp.zeros(1, dtype=jnp.int32)

with open(fit_path) as f:
    results = json.load(f)
    if "subject" not in results["fits"]:
        results["fits"]["subject"] = results["subject"]
    fits = {k: jnp.array(v) for k, v in results["fits"].items()}

# %%

sim_dataset = simulate_h5_from_h5(
    BaseCMRFactory,
    data,
    connections,
    fits,
    trial_mask,
    experiment_count=1,
    rng=rng,
)

for key, array in sim_dataset.items():
    print(f"{key:<12}  shape = {tuple(array.shape)}  dtype = {array.dtype}")

# %%

sim_dataset = jit(simulate_h5_from_h5)(
    BaseCMRFactory,
    data,
    connections,
    fits,
    trial_mask,
    experiment_count=1,
    rng=rng,
)

for key, array in sim_dataset.items():
    print(f"{key:<12}  shape = {tuple(array.shape)}  dtype = {array.dtype}")


# %%

#!%timeit -n 1 -r 1 simulate_h5_from_h5(BaseCMRFactory, data, connections, fits, trial_mask, 1, rng)

# %%

#!%load_ext line_profiler

# %%

#!%lprun -f simulate_h5_from_h5 simulate_h5_from_h5(BaseCMRFactory, data, connections, fits, trial_mask, 1, rng)

# %%
