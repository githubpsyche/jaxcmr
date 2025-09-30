# %% [markdown]
# # Minimal Profiling Script for Fitting Method

# %%
#!%load_ext autoreload
#!%autoreload 2

import os
import json
import warnings

import jax.numpy as jnp
import numpy as np

from jaxcmr.helpers import load_data, generate_trial_mask
from jaxcmr.fitting import ScipyDE as fitting_method
from jaxcmr.models.weird_cmr import BaseCMRFactory as model_factory
from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator as loss_fn_generator

warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Define Parameters and Data Path

# %%
data_path = "data/HealeyKahana2014.h5"  # Adjust if needed
data_query = "data['listtype'] == -1"   # Or your desired filter

# Fitting configuration
model_name = "WeirdBaseCMR"
popsize = 15
num_steps = 1000
relative_tolerance = 0.001
cross_rate = 0.9
diff_w = 0.85
best_of = 1

# Parameter bounds
parameters = {
    "fixed": {},
    "free": {
        "encoding_drift_rate": [1e-16, 1.0],
        "start_drift_rate": [1e-16, 1.0],
        "recall_drift_rate": [1e-16, 1.0],
        "shared_support": [1e-16, 100.0],
        "item_support": [1e-16, 100.0],
        "learning_rate": [1e-16, 1.0],
        "primacy_scale": [1e-16, 100.0],
        "primacy_decay": [1e-16, 100.0],
        "stop_probability_scale": [1e-16, 1.0],
        "stop_probability_growth": [1e-16, 10.0],
        "choice_sensitivity": [1e-16, 100.0],
    },
}

# %% [markdown]
# ## 2. Load Data and Generate Trial Mask

# %%
data = load_data(data_path)                 # Your custom loader
trial_mask = generate_trial_mask(data, data_query)

# Prepare any additional inputs (e.g., a `connections` matrix)
max_size = np.max(data["pres_itemnos"])
connections = jnp.zeros((max_size, max_size))

# %% [markdown]
# ## 3. Set Up the Fitter

# %%
base_params = parameters["fixed"]
bounds = parameters["free"]

fitter = fitting_method(
    data,
    connections,
    base_params,
    model_factory,
    loss_fn_generator,
    hyperparams={
        "num_steps": num_steps,
        "pop_size": popsize,
        "relative_tolerance": relative_tolerance,
        "cross_over_rate": cross_rate,
        "diff_w": diff_w,
        "progress_bar": True,
        "display_iterations": False,
        "bounds": bounds,
        "best_of": best_of,
    },
)

# %% [markdown]
# ## 4. Profile the Fitting Call
# **Note**: `%prun` is an IPython/Jupyter magic command; it will not work in a standard Python script.

# %%
#!%load_ext line_profiler

#!%lprun -f fitter._fit_single_mask fitter._fit_single_mask(trial_mask)

# %%
