# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: jaxcmr
#     language: python
#     name: python3
# ---

# %%
import itertools
import papermill as pm
from tqdm import tqdm

# %%
data_parameters = [
    {"base_data_tag": "CohenKahana2022",
     "trial_query": "data['subject'] != -1"},
]

# %%
handle_repeats = [False]
handle_elis = [False]

# %%
model_parameters = [
    {
        "model_name": "WeirdCMR",
        "model_factory_path": "jaxcmr.models_repfr.weird_cmr.BaseCMRFactory",
        "redo_fits": True,
        "redo_sims": True,
        "redo_figures": True,
        "parameters": {
            "fixed": {},
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
        },
    },
]

# %%
for data_params, model_params, allow_repeated_recalls, allow_elis in tqdm(
    itertools.product(data_parameters, model_parameters, handle_repeats, handle_elis)
):
    
    # configure handling of extra-list intrusions
    base_data_tag = data_params["base_data_tag"]
    base_data_tag += "_withELI" if allow_elis else "_noELI"

    # configure handling of repeated recalls
    if allow_repeated_recalls:
        filter_repeated_recalls = False
        data_tag = f"RepeatedRecalls{base_data_tag}"
        data_path = f"data/RepeatedRecalls{base_data_tag}.h5"
    else:
        filter_repeated_recalls = True
        data_tag = base_data_tag
        data_path = f"data/{base_data_tag}.h5"

    output_path = (
        f"projects/CohenKahana2022/{data_tag}_{model_params['model_name']}_Fitting_No_Control.ipynb"
    )
    print(output_path)
    print(data_params)
    print(model_params)

    pm.execute_notebook(
        "projects/CohenKahana2022/Fitting_No_Control.ipynb",
        output_path,
        autosave_cell_every=180,
        log_output=True,
        parameters={
            "allow_repeated_recalls": allow_repeated_recalls,
            "filter_repeated_recalls": filter_repeated_recalls,
            "data_tag": data_tag,
            "data_path": data_path,
            **model_params,
            **data_params,
        },
    )
