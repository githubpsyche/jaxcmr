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

from jaxcmr.position_scale_cmr import BaseCMRFactory as model_factory
from jaxcmr.experimental.array import to_numba_typed_dict
from jaxcmr.fitting import ScipyDE as fitting_method
from jaxcmr.helpers import generate_trial_mask, import_from_string, load_data
from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator as loss_fn_generator
from jaxcmr.simulation import simulate_h5_from_h5
from jaxcmr.summarize import summarize_parameters

warnings.filterwarnings("ignore")

# %%

single_analysis_paths = [
    "jaxcmr.repcrp.plot_rep_crp",
]
comparison_analysis_paths = [
    # "compmempy.analyses.rpl.plot_spacing",
    "jaxcmr.spc.plot_spc",
    "jaxcmr.crp.plot_crp",
    "jaxcmr.pnr.plot_pnr",
    # "compmempy.analyses.distance_crp.plot_distance_crp",
]
single_analyses = [import_from_string(path) for path in single_analysis_paths]

comparison_analyses = [import_from_string(path) for path in comparison_analysis_paths]

# %%

# data params
data_name = "LohnasKahana2014"
data_query = "data['list_type'] > 0"
data_path = "data/LohnasKahana2014.h5"
run_tag = "full_best_of_3"

# fitting params
redo_fits = False
model_name = "PositionScaleCMR"
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
        # "choice_sensitivity": 1.0,
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
        # "mcf_trace_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
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

# %%

for combined_LT, lt_values in [
    ("1", [1]),
    ("2", [2]),
    ("3", [3]),
    ("4", [4]),
    ("234", [2, 3, 4]),
    ("34", [3, 4]),
]:
    for analysis in comparison_analyses:
        figure_str = f"{results['name']}_LT{combined_LT}_{analysis.__name__[5:]}.png"
        figure_path = os.path.join("figures/fits/", figure_str)
        print(figure_str)
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

        # Create a mask for data using np.isin for the selected list types
        lt_trial_mask = np.isin(data["list_type"].flatten(), lt_values)
        joint_trial_mask = np.logical_and(trial_mask, lt_trial_mask)

        # Create a mask for simulation data similarly
        _trial_mask = generate_trial_mask(sim, data_query)
        _lt_trial_mask = np.isin(sim["list_type"].flatten(), lt_values)
        _joint_trial_mask = np.logical_and(_trial_mask, _lt_trial_mask)

        axis = analysis(
            datasets=[
                to_numba_typed_dict({key: np.array(val) for key, val in sim.items()}),
                to_numba_typed_dict({key: np.array(val) for key, val in data.items()}),
            ],
            trial_masks=[np.array(_joint_trial_mask), np.array(joint_trial_mask)],
            color_cycle=color_cycle,
            labels=["Model", "Data"],
            contrast_name="source",
            axis=None,
            distances=1 - connections,
        )

        axis.tick_params(labelsize=14)
        axis.set_xlabel(axis.get_xlabel(), fontsize=16)
        axis.set_ylabel(axis.get_ylabel(), fontsize=16)
        plt.savefig(figure_path, bbox_inches="tight", dpi=600)

# %%

for combined_LT, lt_values in [
    ("3", [3]),
    ("4", [4]),
    ("34", [3, 4]),
]:
    for dataset_index, dataset in enumerate([
        to_numba_typed_dict({key: np.array(val) for key, val in sim.items()}),
        # to_numba_typed_dict({key: np.array(value) for key, value in data.items()}),
    ]):
        for analysis in single_analyses:
            figure_str = f"{results['name']}_LT{combined_LT}_{analysis.__name__[5:]}.png"
            figure_path = os.path.join("figures/fits/", figure_str)
            print(figure_str)
            color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

            # Create a mask for simulation data similarly
            _trial_mask = generate_trial_mask(dataset, data_query)
            _lt_trial_mask = np.isin(dataset["list_type"].flatten(), lt_values)
            _joint_trial_mask = np.logical_and(_trial_mask, _lt_trial_mask)

            axis = analysis(
                datasets=[dataset],
                trial_masks=[np.array(_joint_trial_mask)],
                color_cycle=color_cycle,
                labels=["Model", "Data"][dataset_index],
                contrast_name="source",
                axis=None,
                distances=1 - connections,
            )

            axis.tick_params(labelsize=14)
            axis.set_xlabel(axis.get_xlabel(), fontsize=16)
            axis.set_ylabel(axis.get_ylabel(), fontsize=16)
            plt.savefig(figure_path, bbox_inches="tight", dpi=600)
# %%
