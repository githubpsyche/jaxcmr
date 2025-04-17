# %%

#!%load_ext autoreload
#!%autoreload 2

import json
import os
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib import rcParams  # type: ignore

from jaxcmr.experimental.array import to_numba_typed_dict
from jaxcmr.fitting import ScipyDE as fitting_method
from jaxcmr.helpers import (
    generate_trial_mask,
    import_from_string,
    load_data,
    save_dict_to_hdf5,
)
from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator as loss_fn_generator
from jaxcmr.simulation import simulate_h5_from_h5
from jaxcmr.summarize import summarize_parameters
from jaxcmr.weird_position_scale_cmr import BaseCMRFactory as model_factory

warnings.filterwarnings("ignore")

# %%

single_analysis_paths = [
    # "jaxcmr.repcrp.plot_first_rep_crp"
]
comparison_analysis_paths = [
    # "compmempy.analyses.rpl.plot_spacing",
    "jaxcmr.spc.plot_spc",
    # "jaxcmr.repcrp.plot_second_rep_crp"
    "jaxcmr.srac.plot_srac",
    "jaxcmr.crp.plot_crp",
    "jaxcmr.pnr.plot_pnr",
    # "compmempy.analyses.distance_crp.plot_distance_crp",
]
single_analyses = [import_from_string(path) for path in single_analysis_paths]

comparison_analyses = [import_from_string(path) for path in comparison_analysis_paths]

# %%

# data params
data_name = "GordonRanschburg2021"
data_query = "data['condition'] == 2"
data_path = "data/GordonRanschburg2021.h5"
run_tag = "full_best_of_3"

# fitting params
redo_fits = False
model_name = "WeirdPositionScaleCMR"
relative_tolerance = 0.001
popsize = 15
num_steps = 1000
cross_rate = 0.9
diff_w = 0.85
best_of = 3

# sim params
experiment_count = 50
seed = 0

parameters = {
    "fixed": {
        "allow_repeated_recalls": True,
    },
    "free": {
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
    },
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


# add subdirectories for each product type: json, figures, h5
product_dirs = {}
for product in ["fits", "figures"]:  # , "simulations"]:
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
    base_params = parameters["fixed"]
    bounds = parameters["free"]
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
            "best_of": best_of,
        },
    )

    results = fitter.fit(trial_mask)
    results = dict(results)

    with open(fit_path, "w") as f:
        json.dump(results, f, indent=4)

results["data_query"] = data_query
results["model"] = model_name
results["name"] = f"{data_name}_{model_name}_{run_tag}"

with open(fit_path, "w") as f:
    json.dump(results, f, indent=4)

print(
    summarize_parameters([results], query_parameters, include_std=True, include_ci=True)
)

# %%

rng = random.PRNGKey(seed)
rng, rng_iter = random.split(rng)
sim = simulate_h5_from_h5(
    model_factory=model_factory,
    dataset=data,
    connections=connections,
    parameters={key: jnp.array(val) for key, val in results["fits"].items()},  # type: ignore
    trial_mask=trial_mask,
    experiment_count=experiment_count,
    rng=rng_iter,
)

save_dict_to_hdf5(sim, f"fits/{results['name']}.h5")
