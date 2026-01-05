# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pooled Model Comparison Across Datasets
#
# This notebook pools model comparison results across multiple datasets using:
# 1. Summed AIC (total AIC across all subjects in pooled datasets)
# 2. Rank-based aggregation (mean rank across datasets)
# 3. Meta-analytic pooling of ΔAIC (inverse-variance weighted)

# %%
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from jaxcmr.helpers import find_project_root

# %%
# Configuration
fit_tag = "rerun_best_of_1"
fit_dir = "projects/repfr/results/fits/"

# Dataset groupings
free_recall_datasets = ["LohnasKahana2014", "Lohnas2025"]
serial_recall_datasets = ["RepeatedRecallsGordonRanschburg2021", "RepeatedRecallsKahanaJacobs2000"]
all_datasets = free_recall_datasets + serial_recall_datasets

model_names = [
    "WeirdCMRNoStop",
    "NoReinstateCMRNoStop",
    "DistinctContextsCMRNoStop",
    "BasePositionalCMRNoStop",
    "FullPositionalCMRNoStop",
    "McfReinfPositionalCMRNoStop",
    "MfcReinfPositionalCMRNoStop",
    "FullReinfPositionalCMRNoStop",
    "SimpleFullReinfPositionalCMRNoStop",
    "BlendPositionalCMRNoStop",
]

model_titles = [
    "WeirdCMR",
    "NoReinstateCMR",
    "DistinctContextsCMR",
    "BasePositionalCMR",
    "FullPositionalCMR",
    "MCFReinfPositionalCMR",
    "MFCReinfPositionalCMR",
    "FullReinfPositionalCMR",
    "SimpleFullReinfPositionalCMR",
    "BlendPositionalCMR",
]

# %%
def load_results(data_name, model_name):
    """Load fitting results for a dataset-model pair."""
    fit_path = os.path.join(
        find_project_root(), fit_dir,
        f"{data_name}_{model_name}_{fit_tag}.json"
    )
    with open(fit_path) as f:
        result = json.load(f)
    return result

def get_aic_per_subject(result):
    """Compute AIC for each subject: AIC = 2k + 2*NLL."""
    fitness = np.array(result["fitness"])  # NLL per subject (top-level key)
    k = len(result["free"])  # number of free parameters
    return 2 * k + 2 * fitness

def get_total_aic(result):
    """Compute total AIC across all subjects."""
    return np.sum(get_aic_per_subject(result))

def get_mean_aic(result):
    """Compute mean AIC per subject."""
    return np.mean(get_aic_per_subject(result))

# %%
# Load all results
all_results = {}
for data_name in all_datasets:
    all_results[data_name] = {}
    for model_name, model_title in zip(model_names, model_titles):
        all_results[data_name][model_title] = load_results(data_name, model_name)

# %% [markdown]
# ## Method 1: Summed AIC Across Datasets

# %%
def compute_summed_aic(datasets, results_dict):
    """Compute total summed AIC across specified datasets for each model."""
    summed_aic = {}
    for model_title in model_titles:
        total = 0
        for data_name in datasets:
            total += get_total_aic(results_dict[data_name][model_title])
        summed_aic[model_title] = total
    return summed_aic

def summed_aic_table(datasets, results_dict, label=""):
    """Create a table of summed AIC results."""
    summed = compute_summed_aic(datasets, results_dict)
    df = pd.DataFrame({
        "Model": list(summed.keys()),
        "Summed AIC": list(summed.values())
    })
    df = df.sort_values("Summed AIC").reset_index(drop=True)
    min_aic = df["Summed AIC"].min()
    df["ΔAIC"] = df["Summed AIC"] - min_aic
    df["Relative Likelihood"] = np.exp(-0.5 * df["ΔAIC"])
    df["AIC Weight"] = df["Relative Likelihood"] / df["Relative Likelihood"].sum()
    return df

# %%
print("=" * 60)
print("SUMMED AIC: FREE RECALL DATASETS")
print("=" * 60)
print(f"Datasets: {free_recall_datasets}\n")
df_free = summed_aic_table(free_recall_datasets, all_results)
print(df_free.to_string(index=False))

# %%
print("=" * 60)
print("SUMMED AIC: SERIAL RECALL DATASETS")
print("=" * 60)
print(f"Datasets: {serial_recall_datasets}\n")
df_serial = summed_aic_table(serial_recall_datasets, all_results)
print(df_serial.to_string(index=False))

# %%
print("=" * 60)
print("SUMMED AIC: ALL DATASETS")
print("=" * 60)
print(f"Datasets: {all_datasets}\n")
df_all = summed_aic_table(all_datasets, all_results)
print(df_all.to_string(index=False))

# %% [markdown]
# ## Method 2: Rank-Based Aggregation

# %%
def compute_ranks_per_dataset(results_dict):
    """Compute AIC-based ranks for each model within each dataset."""
    ranks = {}
    for data_name in all_datasets:
        # Get mean AIC per model for this dataset
        model_aic = {
            model_title: get_mean_aic(results_dict[data_name][model_title])
            for model_title in model_titles
        }
        # Convert to ranks (1 = best)
        sorted_models = sorted(model_aic.keys(), key=lambda x: model_aic[x])
        ranks[data_name] = {model: rank + 1 for rank, model in enumerate(sorted_models)}
    return ranks

def rank_aggregation_table(results_dict):
    """Create a table showing ranks across datasets and mean rank."""
    ranks = compute_ranks_per_dataset(results_dict)

    data = []
    for model_title in model_titles:
        row = {"Model": model_title}
        for data_name in all_datasets:
            short_name = data_name.replace("RepeatedRecalls", "").replace("Kahana", "K").replace("Gordon", "G").replace("Ranschburg", "R")
            row[short_name] = ranks[data_name][model_title]
        row["Mean Rank"] = np.mean([ranks[dn][model_title] for dn in all_datasets])
        row["Mean Rank (FR)"] = np.mean([ranks[dn][model_title] for dn in free_recall_datasets])
        row["Mean Rank (SR)"] = np.mean([ranks[dn][model_title] for dn in serial_recall_datasets])
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values("Mean Rank").reset_index(drop=True)
    return df

# %%
print("=" * 60)
print("RANK-BASED AGGREGATION")
print("=" * 60)
df_ranks = rank_aggregation_table(all_results)
print(df_ranks.to_string(index=False))

# %% [markdown]
# ## Method 3: Meta-Analytic Pooling of ΔAIC

# %%
def compute_pairwise_delta_aic_stats(results_dict, data_name, ref_model):
    """Compute mean ΔAIC and SE for each model vs reference model."""
    ref_aic = get_aic_per_subject(results_dict[data_name][ref_model])

    stats_dict = {}
    for model_title in model_titles:
        if model_title == ref_model:
            stats_dict[model_title] = {"mean": 0, "se": 0, "n": len(ref_aic)}
            continue
        model_aic = get_aic_per_subject(results_dict[data_name][model_title])
        delta = model_aic - ref_aic  # positive = model is worse
        stats_dict[model_title] = {
            "mean": np.mean(delta),
            "se": np.std(delta, ddof=1) / np.sqrt(len(delta)),
            "n": len(delta)
        }
    return stats_dict

def meta_analytic_pooling(results_dict, datasets, ref_model):
    """Pool ΔAIC estimates across datasets using inverse-variance weighting."""
    pooled = {}

    for model_title in model_titles:
        if model_title == ref_model:
            pooled[model_title] = {"pooled_mean": 0, "pooled_se": 0, "ci_lower": 0, "ci_upper": 0}
            continue

        means = []
        weights = []

        for data_name in datasets:
            stats = compute_pairwise_delta_aic_stats(results_dict, data_name, ref_model)
            if stats[model_title]["se"] > 0:
                means.append(stats[model_title]["mean"])
                weights.append(1 / (stats[model_title]["se"] ** 2))

        if len(weights) > 0:
            weights = np.array(weights)
            means = np.array(means)
            pooled_mean = np.sum(weights * means) / np.sum(weights)
            pooled_se = np.sqrt(1 / np.sum(weights))
            ci_lower = pooled_mean - 1.96 * pooled_se
            ci_upper = pooled_mean + 1.96 * pooled_se
        else:
            pooled_mean = pooled_se = ci_lower = ci_upper = np.nan

        pooled[model_title] = {
            "pooled_mean": pooled_mean,
            "pooled_se": pooled_se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    return pooled

def meta_analysis_table(results_dict, datasets, label=""):
    """Create meta-analysis table with best model as reference."""
    # First find the best model by summed AIC
    summed = compute_summed_aic(datasets, results_dict)
    best_model = min(summed.keys(), key=lambda x: summed[x])

    pooled = meta_analytic_pooling(results_dict, datasets, best_model)

    data = []
    for model_title in model_titles:
        p = pooled[model_title]
        reliable = "Yes" if p["ci_lower"] > 0 or p["ci_upper"] < 0 else "No"
        if model_title == best_model:
            reliable = "-"
        data.append({
            "Model": model_title,
            "Pooled ΔAIC": f"{p['pooled_mean']:.2f}",
            "95% CI": f"[{p['ci_lower']:.2f}, {p['ci_upper']:.2f}]",
            "Reliably Worse?": reliable
        })

    df = pd.DataFrame(data)
    # Sort by pooled ΔAIC
    df["_sort"] = [pooled[m]["pooled_mean"] for m in df["Model"]]
    df = df.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)

    print(f"Reference model (best by summed AIC): {best_model}\n")
    return df

# %%
print("=" * 60)
print("META-ANALYTIC POOLING: FREE RECALL")
print("=" * 60)
df_meta_fr = meta_analysis_table(all_results, free_recall_datasets)
print(df_meta_fr.to_string(index=False))

# %%
print("=" * 60)
print("META-ANALYTIC POOLING: SERIAL RECALL")
print("=" * 60)
df_meta_sr = meta_analysis_table(all_results, serial_recall_datasets)
print(df_meta_sr.to_string(index=False))

# %%
print("=" * 60)
print("META-ANALYTIC POOLING: ALL DATASETS")
print("=" * 60)
df_meta_all = meta_analysis_table(all_results, all_datasets)
print(df_meta_all.to_string(index=False))

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("SUMMARY: BEST MODELS BY POOLING METHOD")
print("=" * 60)
print()
print("FREE RECALL:")
print(f"  Summed AIC winner:     {df_free.iloc[0]['Model']}")
print(f"  Mean rank winner:      {df_ranks.sort_values('Mean Rank (FR)').iloc[0]['Model']}")
print()
print("SERIAL RECALL:")
print(f"  Summed AIC winner:     {df_serial.iloc[0]['Model']}")
print(f"  Mean rank winner:      {df_ranks.sort_values('Mean Rank (SR)').iloc[0]['Model']}")
print()
print("ALL DATASETS:")
print(f"  Summed AIC winner:     {df_all.iloc[0]['Model']}")
print(f"  Mean rank winner:      {df_ranks.sort_values('Mean Rank').iloc[0]['Model']}")
