# %%

#!%load_ext autoreload
#!%autoreload 2

import json
import os
import warnings
from typing import Optional

import h5py
import jax.numpy as jnp
import numpy as np
from jax import random

from jaxcmr.cmr import CMRFactory as model_factory
from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator as loss_fn_generator
from jaxcmr.fitting import ScipyDE as fitting_method
from jaxcmr.simulation import simulate_h5_from_h5
from jaxcmr.summarize import summarize_parameters
from jaxcmr.typing import Array, Bool

warnings.filterwarnings("ignore")

# %%

# data params
data_name = "HealyKahana2014"
data_query = "data['listtype'] == -1"
data_path = "data/HealyKahana2014.h5"

# fitting params
redo_fits = True
run_tag = "test"
model_name = "BaseCMR"
relative_tolerance = 0.001
popsize = 15
num_steps = 1000
cross_rate = 0.9
diff_w = 0.85

# sim params
experiment_count = 50
seed = 0

base_params = {
    "encoding_drift_rate": 0.999,
    "start_drift_rate": 0.0001,
    "recall_drift_rate": 0.999,
    "shared_support": 0.0,
    "item_support": 0.0,
    "learning_rate": 0.999,
    "primacy_scale": 0.0,
    "primacy_decay": 0.0,
    "stop_probability_scale": 1.0,
    "stop_probability_growth": 1.0,
    "choice_sensitivity": 1.0,
}
bounds = {
    "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
    "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
    "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
    "shared_support": [2.220446049250313e-16, 99.9999999999999998],
    "item_support": [2.220446049250313e-16, 99.9999999999999998],
    "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
    "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
    "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
    "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
    "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
    "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
}

query_parameters = [
    "encoding_drift_rate",
    "start_drift_rate",
    "recall_drift_rate",
    "shared_support",
    "item_support",
    "learning_rate",
    "primacy_scale",
    "primacy_decay",
    "mcf_trace_sensitivity",
    "stop_probability_scale",
    "stop_probability_growth",
    "choice_sensitivity",
]

# %%


def load_data(data_path: str) -> dict[str, jnp.ndarray]:
    """Load data from hdf5 file."""
    with h5py.File(data_path, "r") as f:
        result = {key: f["/data"][key][()].T for key in f["/data"].keys()}  # type: ignore
    return {key: jnp.array(value) for key, value in result.items()}


def generate_trial_mask(
    data: dict, trial_query: Optional[str]
) -> Bool[Array, " trial_count"]:
    """Returns a boolean mask for selecting trials based on a specified query condition.

    Args:
        data: dict containing trial data arrays, including a "recalls" key with an array.
        trial_query: condition to evaluate, which should return a boolean array.
        If None, returns a mask that selects all trials.
    """
    if trial_query is None:
        return jnp.ones(data["recalls"].shape[0], dtype=bool)
    return eval(trial_query).flatten()


# add subdirectories for each product type: json, figures, h5
product_dirs = {}
for product in ["fits", "figures", "simulations"]:
    product_dir = os.path.join(product)
    product_dirs[product] = product_dir
    if not os.path.exists(product_dir):
        os.makedirs(product_dir)

data = load_data(data_path)
trial_mask = generate_trial_mask(data, data_query)

max_size = np.max(data["pres_itemnos"])
connections = jnp.zeros((max_size, max_size))

# %%

fit_path = os.path.join(
    product_dirs["fits"], f"{data_name}_{model_name}_{run_tag}.json"
)
print(fit_path)

if os.path.exists(fit_path) and not redo_fits:
    with open(fit_path) as f:
        results = json.load(f)
        if "subject" not in results["fits"]:
            results["fits"]["subject"] = results["subject"]

else:
    fitter = fitting_method(
        {key: jnp.array(value) for key, value in data.items()},
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
        },
    )

    results = fitter.fit(trial_mask)
    results = dict(results)

    with open(fit_path, "w") as f:
        json.dump(results, f, indent=4)

results["data_query"] = data_query
results["model"] = model_name
results["name"] = f"{data_name}_{model_name}_{run_tag}"
results["relative_tolerance"] = relative_tolerance
results["popsize"] = popsize
results["num_steps"] = num_steps
results["cross_rate"] = cross_rate
results["diff_w"] = diff_w

print(
    summarize_parameters([results], query_parameters, include_std=True, include_ci=True)
)

# %%

rng = random.PRNGKey(seed)
rng, rng_iter = random.split(rng)
simulate_h5_from_h5(
    model_factory=model_factory,
    dataset=data,
    connections=connections,
    parameters={key: jnp.array(val) for key, val in results["fits"].items()},  # type: ignore
    trial_mask=trial_mask,
    experiment_count=experiment_count,
    rng=rng_iter,
)
