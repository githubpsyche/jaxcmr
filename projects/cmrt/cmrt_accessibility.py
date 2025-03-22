# %%
#!%load_ext autoreload
#!%autoreload 2

# %%
# | parametrization

import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import jit, lax, vmap
from tqdm import trange

from jaxcmr.cmr import BaseCMRFactory as model_factory
from jaxcmr.fitting import make_subject_trial_masks
from jaxcmr.helpers import generate_trial_mask, load_data
from jaxcmr.likelihood import MemorySearchLikelihoodFnGenerator as LikelihoodFnGenerator

# %%
# | CONFIG / PARAMETERS

fit_path = "fits/Murdock1962_BaseCMR_full.json"
data_path = "data/Murdock1962.h5"
data_query = "data['subject'] > -1"

use_progress_bar = True

# %%
# | CONFIG / PARAMETERS

fit_path = "fits/HerremaKahana2024_BaseCMR_full.json"
data_path = "data/HerremaKahana2024.h5"
data_query = "data['subject'] > -1"

use_progress_bar = True

# %%
# | LOAD FIT RESULTS & DATA

with open(fit_path) as f:
    fit_result = json.load(f)
    if "subject" not in fit_result["fits"]:
        fit_result["fits"]["subject"] = fit_result["subject"]


data = load_data(data_path)
trial_mask = generate_trial_mask(data, data_query)
max_size = np.max(data["pres_itemnos"])
connections = jnp.zeros((max_size, max_size))

likelihood_generator = LikelihoodFnGenerator(model_factory, data, connections)

subjects = data["subject"].flatten()
subject_trial_masks, unique_subjects = make_subject_trial_masks(trial_mask, subjects)
subject_range = (
    trange(len(unique_subjects)) if use_progress_bar else range(len(unique_subjects))
)

# count how many unique list lengths there are per subject
list_lengths = np.array(
    [len(np.unique(data["listLength"][data["subject"] == s])) for s in unique_subjects]
)
list_lengths

# %%
# | helper functions


@jit
def get_accessibility(trial_index, parameters):
    _, prescaled_activations = lax.scan(
        lambda m, c: (
            m.retrieve(c),
            jnp.dot(m.context.state, m.mcf.state) * m.recallable,
        ),
        likelihood_generator.init_model_for_retrieval(trial_index, parameters),
        likelihood_generator.trials[trial_index],
    )

    # converted to an average activation over recallable items at each retrieval attempt
    # return jnp.sum(prescaled_activations, axis=1) / jnp.sum(
    # prescaled_activations != 0, axis=1
    # )

    # converted to a maximum activation over recallable items at each retrieval attempt
    # return jnp.max(prescaled_activations, axis=1)

    # converted to a sum of activations over recallable items at each retrieval attempt
    return jnp.sum(prescaled_activations, axis=1)


@jit
def get_termination_index(trial_index):
    return jnp.sum(likelihood_generator.trials[trial_index] != 0)


@jit
def get_recall_positions(trial_index):
    return jnp.arange(1, len(likelihood_generator.trials[trial_index]) + 1)


@jit
def get_list_length(trial_index):
    return (likelihood_generator.present_lists[trial_index] != 0).sum()


@jit
def get_reaction_times(trial_index):
    return data["irt"][trial_index]


# %%

all_rows = []

for s in subject_range:
    if np.sum(subject_trial_masks[s]) == 0:
        continue

    trial_indices = jnp.where(subject_trial_masks[s])[0]
    parameters = {
        key: fit_result["fits"][key][s]
        for key in fit_result["fits"]
        if key != "subject"
    }

    # extract trial information
    trials = likelihood_generator.trials[trial_indices]
    accessibility = vmap(get_accessibility, in_axes=(0, None))(
        trial_indices, parameters
    )
    termination_indices = vmap(get_termination_index)(trial_indices)
    recall_positions = vmap(get_recall_positions)(trial_indices)
    list_lengths = vmap(get_list_length)(trial_indices)
    reaction_times = vmap(get_reaction_times)(trial_indices)

    for i, trial_index in enumerate(trial_indices):
        for j in range(termination_indices[i] + 1):
            continuation = j < termination_indices[i]
            rt = int(reaction_times[i][j].item())
            log_rt = np.log(rt) if rt != 0 else 0
            row = {
                "subject": int(unique_subjects[s]),
                "trial_index": int(trial_index.item()),
                "recall": int(trials[i][j].item()),
                "recall_position": int(recall_positions[i][j].item()),
                "list_length": int(list_lengths[i].item()),
                "accessibility": float(accessibility[i][j].item()),
                "continuation": "Continued"
                if j < termination_indices[i].item()
                else "Terminated",
                "reaction_time": rt,
                "log_reaction_time": float(log_rt),
            }
            all_rows.append(row)

all_rows = pd.DataFrame(all_rows)
all_rows.head()

# %%
df = all_rows

# use pivot_table to unify by subject
# df = df.pivot_table(
#     index=["subject", "list_length", "recall_position", "continuation"],
#     values=["accessibility", "reaction_time"],
#     aggfunc="mean",
# ).reset_index()

unique_list_lengths = df["list_length"].unique()
for ll in sorted(unique_list_lengths):
    subset = df[df["list_length"] == ll]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset,
        x="recall_position",
        y="accessibility",
        hue="continuation",
        errorbar="ci",
    )

    plt.xlabel("Recall Position")
    plt.ylabel("Mean Accessibility")
    plt.title(f"Mean Accessibility by Recall Position (List Length = {ll})")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(title="Recall Status")
    plt.show()
# %%

plt.figure(figsize=(8, 6))
subset = all_rows[all_rows["reaction_time"] > 0]

sns.regplot(data=subset, x="accessibility", y="reaction_time", scatter_kws={"s": 10})
plt.xlabel("Mean Accessibility")
plt.ylabel("Reaction Time")
plt.title("Relationship between Accessibility and Reaction Time")
plt.show()

# %%

subset = all_rows[all_rows["reaction_time"] > 0]

plt.figure(figsize=(8, 6))
sns.regplot(
    data=subset, x="accessibility", y="log_reaction_time", scatter_kws={"s": 10}
)
plt.xlabel("Mean Accessibility")
plt.ylabel("Log(Reaction Time)")
plt.title("Relationship between Accessibility and Log(Reaction Time)")
plt.show()

# %%
# Bin recall positions into 5 groups.
all_rows["recall_bin"] = pd.cut(all_rows["recall_position"], bins=5)
subset = all_rows[all_rows["reaction_time"] > 0]

unique_list_lengths = subset["list_length"].unique()
for ll in sorted(unique_list_lengths):
    ll_subset = subset[subset["list_length"] == ll]
    sns.lmplot(
        data=ll_subset,
        x="accessibility",
        y="reaction_time",
        hue="recall_bin",
        ci=95,
        scatter_kws={"s": 10},
    )
    plt.xlabel("Mean Accessibility")
    plt.ylabel("Reaction Time")
    plt.title(
        "Relationship between Accessibility and Reaction Time\n(Controlled by Recall Position Bin)"
    )
    plt.show()

# %%
# Bin recall positions into 5 groups.
all_rows["recall_bin"] = pd.cut(all_rows["recall_position"], bins=5)
subset = all_rows[all_rows["reaction_time"] > 0]

unique_list_lengths = subset["list_length"].unique()
for ll in sorted(unique_list_lengths):
    ll_subset = subset[subset["list_length"] == ll]
    sns.lmplot(
        data=ll_subset,
        x="accessibility",
        y="log_reaction_time",
        hue="recall_bin",
        ci=95,
        scatter_kws={"s": 10},
    )
    plt.xlabel("Mean Accessibility")
    plt.ylabel("Log(Reaction Time)")
    plt.title(
        "Relationship between Accessibility and Log(Reaction Time)\n(Controlled by Recall Position Bin)"
    )
    plt.show()

# %%

# %%
