# %% [markdown]
# # Comparing MixedCMRFactory Implementations (Single-Subject LL Check)
#
# In this literate-programming notebook (using `py:percent` cells), we'll:
#
# 1. Load a recall dataset (e.g., Healey & Kahana 2014 data).
# 2. Load a fitted parameter set (from a JSON file).
# 3. Compare two different `MixedCMRFactory` implementations:
#    - **Factory A**: `from jaxcmr.cmr import MixedCMRFactory`
#    - **Factory B**: `from jaxcmr.instance_cmr import MixedCMRFactory`
#
# We'll do the comparison for **just the first subject** found in the dataset/fits, 
# computing and comparing a single log-likelihood (LL) value from each factory.

# %%
import os
import json
import numpy as np
import jax.numpy as jnp

# Factory A
from jaxcmr.cmr import MixedCMRFactory as MixedCMRFactoryA

# Factory B
from jaxcmr.instance_cmr import MixedCMRFactory as MixedCMRFactoryB

from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator
from jaxcmr.helpers import load_data
from jax import numpy as jnp

# %% [markdown]
# ## 1. Load the Dataset

# %%
data_path = "data/HealeyKahana2014.h5"
data = load_data(data_path)

print("Dataset keys:", list(data.keys()))
print("Number of trials:", data["recalls"].shape[0])

# %% [markdown]
# ## 2. Load the Fit Results
#
# We'll load a JSON file containing the optimized parameters. 
# This could be a multi-subject fit or a single-subject fit.

# %%
fit_results_path = os.path.join("fits", "HealeyKahana2014_InstanceCMR_best_of_1.json")
with open(fit_results_path, "r") as f:
    fit_results = json.load(f)

print("Fit results keys:", list(fit_results.keys()))
print("Parameter names in fit:", fit_results["fits"].keys())

# %% [markdown]
# ## 3. Select the First Subject
#
# We assume there's at least one subject. If it's a multi-subject fit, we'll just pick the first subject ID 
# and the corresponding parameter values.

# %%
all_subjects = np.unique(data["subject"].flatten())
subject_id = all_subjects[0]
print(f"Using subject {subject_id}")

# Create a boolean mask for trials belonging to this subject
trial_mask = (data["subject"].flatten() == subject_id)

print(f"Number of trials for subject {subject_id}: {trial_mask.sum()}")

# %% [markdown]
# ## 4. Prepare a Connectivity Matrix (If Needed)
#
# If no semantic or associative connections are used, we can just supply a zero matrix of the appropriate size.
# We'll do that here for simplicity.

# %%
max_itemno = np.max(data["pres_itemnos"])
connections = jnp.zeros((max_itemno, max_itemno))

# %% [markdown]
# ## 5. Extract Parameters for this Subject
#
# If the fit is multi-subject, each parameter array in `fit_results["fits"]` will have one entry per subject. 
# We'll extract the entry that corresponds to our `subject_id`.
#
# If the fit is single-subject (or a single global fit), we might only have scalars or length-1 arrays. 
# We'll handle that by just indexing safely.

# %%
fit_dict = fit_results["fits"]
param_names = [k for k in fit_dict.keys() if k not in ("subject",)]

# First, see if "subject" is present in the fits and if its length matches `all_subjects`
# If so, we assume multi-subject fits. Otherwise, single-subject.
is_multisubject = ("subject" in fit_dict) and (len(fit_dict["subject"]) == len(all_subjects))

# Find which index in fit_dict["subject"] corresponds to our chosen subject_id
if is_multisubject:
    # We assume the order in fit_dict["subject"] matches `all_subjects`
    subject_index = np.where(np.array(fit_dict["subject"]) == subject_id)[0][0]
    print(f"Subject index in the fit arrays: {subject_index}")
else:
    subject_index = None

# Build a simple dictionary of param_name -> single float value for the chosen subject
params_for_subject = {}
for p in param_names:
    arr = np.array(fit_dict[p], dtype=float)
    if is_multisubject:
        value = arr[subject_index]
    else:
        # Single-subject or single global fit => just use the (0)-th or scalar
        value = arr[0] if arr.ndim == 1 else float(arr)
    params_for_subject[p] = float(value)

print("Subject parameter dictionary:")
params_for_subject

# %% [markdown]
# ## 6. Compute LL with Factory A vs. Factory B
#
# We'll define a helper function to compute the negative log-likelihood (NLL) for this subject.

# %%
def compute_nll_for_subject(
    data_dict: dict[str, np.ndarray],
    connections: jnp.ndarray,
    trial_mask: np.ndarray,
    subject_params: dict[str, float],
    model_factory,
) -> float:
    """
    Compute negative LL for a single subject given their trial_mask and subject_params,
    using the specified `model_factory`.
    """
    # Convert boolean trial_mask to integer indices for JAX
    trial_indices = jnp.where(trial_mask, size=trial_mask.size)[0]

    # Set up the likelihood function generator
    generator = MemorySearchLikelihoodFnGenerator(model_factory, data_dict, connections)

    # Decide whether to use the "base" or "present_and_predict" approach
    # based on whether all pres_itemnos are identical for this subject's trials.
    these_pres = data_dict["pres_itemnos"][trial_mask]
    # if all rows identical:
    if np.all(np.all(these_pres[0] == these_pres, axis=1)):
        nll = generator.base_predict_trials_loss(trial_indices, subject_params)
    else:
        nll = generator.present_and_predict_trials_loss(trial_indices, subject_params)

    return float(nll)

# %%
nll_A = compute_nll_for_subject(
    data_dict=data,
    connections=connections,
    trial_mask=trial_mask,
    subject_params=params_for_subject,
    model_factory=MixedCMRFactoryA,
)

nll_B = compute_nll_for_subject(
    data_dict=data,
    connections=connections,
    trial_mask=trial_mask,
    subject_params=params_for_subject,
    model_factory=MixedCMRFactoryB,
)

# %% [markdown]
# ## 7. Compare the Two LL Values
#
# We'll just print them side-by-side and see if they match (to within floating-point tolerance).

# %%
print(f"Subject {subject_id} LL comparison:")
print(f"  Factory A: NLL = {nll_A:.6f}")
print(f"  Factory B: NLL = {nll_B:.6f}")
diff = abs(nll_A - nll_B)
print(f"  Absolute difference = {diff:.2e}")

# %% [markdown]
# **Conclusion**: If both factory implementations are identical (just housed in different modules), 
# the negative log-likelihood values should be extremely close (differing at most by small numerical round-off). 
# A difference near zero (e.g., < 1e-12) indicates they produce the same LL for this subject. 