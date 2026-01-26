# %% [markdown]
# Parameter shifting: boosted end-of-list learning with post-list drift
#
# This notebook-style script mirrors the structure of `templates/parameter_shifting.ipynb`.
# It samples subject-level fits, boosts learning for the last items in a study list,
# drifts context out-of-list, and tracks how a boost parameter shifts recall-related summaries.

# %%
import json
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax, random

from jaxcmr.models.cmr import CMR

warnings.filterwarnings("ignore")

# %% [markdown]
# Parameters

# %% tags=["parameters"]
# Run configuration
base_run_tag = "spc_mse_fixed_term"
experiment_count = 500
max_subjects = 0
best_of = 3

# Data parameters
base_data_tag = "TalmiEEG"
data_tag = "TalmiEEG"
data_path = "data/TalmiEEG.h5"
embedding_path = ""  # "data/peers-all-mpnet-base-v2.npy"
emotion_feature_path = ""  # "data/emotion_features_7col.npy"
feature_column = 6
concat_features = False
trial_query = "data['subject'] > -1"
target_directory = "results/"

# algorithm selection
model_name = "AttentionSimpleECMRNoStop"
make_factory_path = "jaxcmr.models.attention_simple_ecmr.make_factory"
component_paths = {
    "mfc_create_fn": "jaxcmr.components.linear_memory.init_mfc",
    "mcf_create_fn": "jaxcmr.components.linear_memory.init_mcf",
    "context_create_fn": "jaxcmr.components.context.init",
    "termination_policy_create_fn": "jaxcmr.components.termination.NoStopTermination",
}

# Parameter sweep placeholder
varied_parameter = "emotion_scale"
sweep_min = 0.0
sweep_max = 10.0

# Flow toggles
filter_repeated_recalls = True
handle_elis = False
redo_fits = False
redo_figures = True
redo_sims = False

# analysis configuration placeholder
comparison_analysis_configs = [
    {
        "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
        "figure_suffix": "cat_spc_negative",
        "kwargs": {"category_field": "condition", "category_values": [1]},
    },
    {
        "target": "jaxcmr.analyses.cat_spc.plot_cat_spc",
        "figure_suffix": "cat_spc_neutral",
        "kwargs": {"category_field": "condition", "category_values": [2]},
    },
    {"target": "jaxcmr.analyses.spc.plot_spc"},
    {"target": "jaxcmr.analyses.crp.plot_crp"},
    {"target": "jaxcmr.analyses.pnr.plot_pnr"},
]

# Simulation-specific controls
seed = 0
subject_samples = 20
list_length = 20
boost_count = 4
boost_min = 0.5
boost_max = 2.0
boost_steps = 4  # number of boost levels between min and max (inclusive)
drift_rate = 0.99

# %%
def load_subject_params(
    path: str,
) -> tuple[dict[str, float], dict[str, Sequence[float]]]:
    """Returns fixed and free parameter arrays from a fit file."""
    data = json.loads(Path(path).read_text())
    fixed = data.get("fixed", {})
    free = data.get("free", data.get("best", {}))
    free = {k: v for k, v in free.items() if isinstance(v, list)}
    return fixed, free


def assemble_params(
    fixed: Mapping[str, float], free: Mapping[str, Sequence[float]], idx: int
) -> dict[str, float]:
    """Returns combined parameter dict for a single subject index."""
    params = dict(fixed)
    for name, values in free.items():
        if name == "subject":
            continue
        params[name] = float(values[idx])
    return params


def simulate_boosted_list(
    params: Mapping[str, float],
    list_length: int,
    boost_factor: float,
    boost_count: int,
    drift_rate: float,
) -> dict[str, float]:
    """Returns summary stats after boosted study and post-list drift."""
    model = CMR(list_length, params)
    base_lr = model.mfc_learning_rate
    sequence = jnp.arange(1, list_length + 1, dtype=int)
    split = list_length - boost_count

    def study_step(i: int, state: CMR) -> CMR:
        lr = jnp.where(i >= split, base_lr * boost_factor, base_lr)
        updated = state.replace(mfc_learning_rate=lr)
        return updated.experience(sequence[i])

    studied = lax.fori_loop(0, sequence.size, study_step, model)
    drifted_context = studied.context.integrate(studied.context.outlist_input, drift_rate)
    drifted = studied.replace(context=drifted_context)
    probs = drifted.outcome_probabilities()
    item_probs = probs[1:]
    return {
        "stop_prob": float(probs[0]),
        "first_mean": float(jnp.mean(item_probs[:-boost_count])),
        "last_mean": float(jnp.mean(item_probs[-boost_count:])),
    }


# %% [markdown]
# Simulation: sample subjects, sweep boost levels, collect summaries

# %%
run_tag = f"{base_run_tag}_best_of_{best_of}"
fit_path = Path(target_directory) / "fits" / f"{data_tag}_{model_name}_{run_tag}.json"

fixed, free = load_subject_params(str(fit_path))
subject_total = len(next(iter(free.values())))
rng = random.PRNGKey(seed)
sample_indices = random.randint(rng, (subject_samples,), 0, subject_total)
boost_levels = jnp.linspace(boost_min, boost_max, boost_steps)

results = []
for subj_idx in sample_indices:
    params = assemble_params(fixed, free, int(subj_idx))
    for boost in boost_levels:
        stats = simulate_boosted_list(
            params, list_length, float(boost), boost_count, drift_rate
        )
        results.append(
            {
                "subject_idx": int(subj_idx),
                "boost": float(boost),
                **stats,
            }
        )


# %% [markdown]
# Summary and plots

# %%
def aggregate_mean(key: str, boost: float) -> float:
    values = [r[key] for r in results if r["boost"] == boost]
    return float(jnp.mean(jnp.array(values)))


summary = []
for boost in boost_levels:
    summary.append(
        {
            "boost": float(boost),
            "stop_prob": aggregate_mean("stop_prob", float(boost)),
            "first_mean": aggregate_mean("first_mean", float(boost)),
            "last_mean": aggregate_mean("last_mean", float(boost)),
        }
    )

print("Boost sweep summaries:")
for row in summary:
    print(row)

# Plot summaries similar to parameter_shifting outputs
fig, ax = plt.subplots(figsize=(6, 4))
boost_vals = np.array([row["boost"] for row in summary])
ax.plot(boost_vals, [row["first_mean"] for row in summary], label="First items")
ax.plot(boost_vals, [row["last_mean"] for row in summary], label="Last items")
ax.plot(boost_vals, [row["stop_prob"] for row in summary], label="Stop prob")
ax.set_xlabel("Learning-rate boost (last items)")
ax.set_ylabel("Mean probability")
ax.set_title("Effect of end-of-list boost")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
