"""
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
# ---
"""

# %% [markdown]
# # Repeated‑Recall Audit — Cohen & Kahana (2022)
#
# This percent notebook loads the HDF5 dataset used for fitting, applies a trial
# selection query, and reports whether any trials contain within‑trial repeated
# recalls (ignoring padding zeros). It also counts the total number of repeated
# mentions beyond the first occurrence and can summarize by subject.
#
# Use this to verify that the dataset you fit (e.g., `CohenKahana2022_noELI.h5`)
# has repeats filtered as intended.

# %% [markdown]
# ## Parameters
#
# Adjust these to match the notebook run you want to audit. By default, the
# query mirrors the focal fitting notebook (`session == 1`).

# %%
from __future__ import annotations

DATA_PATH = "data/CohenKahana2022_noELI.h5"
TRIAL_QUERY = "data['session'] == 1"
EXAMPLE_HEAD = 20  # number of example trial indices to preview
SHOW_BY_SUBJECT = True

# %% [markdown]
# ## Imports

# %%
from collections import Counter
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from jaxcmr.helpers import generate_trial_mask, has_repeats_per_row, load_data


# %% [markdown]
# ## Small helpers (local to this notebook)

# %%
def count_repeated_mentions(rows: np.ndarray) -> tuple[int, int]:
    """Returns (trials_with_repeats, total_extra_mentions).

    Args:
      rows: 2D numpy array of recall positions; 1-indexed; 0 for no recall.
    """
    trials_with_repeats = 0
    extra_mentions = 0
    for row in rows:
        nz = row[row > 0]
        if nz.size == 0:
            continue
        c = Counter(nz.tolist())
        extras = sum(max(0, v - 1) for v in c.values())
        if extras > 0:
            trials_with_repeats += 1
            extra_mentions += extras
    return trials_with_repeats, extra_mentions


def preview_list(xs: Iterable[int], limit: int) -> str:
    """Returns a concise preview string for a list of integers.

    Args:
      xs: Values to preview.
      limit: Maximum number of values to include in the preview.
    """
    xs = list(xs)
    if not xs:
        return "[]"
    if len(xs) <= limit:
        return f"[{', '.join(map(str, xs))}]"
    head = ", ".join(map(str, xs[:limit]))
    return f"[{head}, …] (total {len(xs)})"


# %% [markdown]
# ## Load dataset and apply query

# %%
data = load_data(DATA_PATH)
trial_mask = generate_trial_mask(data, TRIAL_QUERY).astype(bool)

recalls = np.array(data["recalls"][trial_mask])
if recalls.size == 0:
    raise SystemExit("No trials selected by the query; nothing to audit.")

selected_indices = np.where(np.array(trial_mask))[0]

print("Dataset:", DATA_PATH)
print("Query:", TRIAL_QUERY)
print("Trials selected:", int(recalls.shape[0]))

# %% [markdown]
# ## Identify repeat‑containing trials
#
# We use `helpers.has_repeats_per_row`, which treats zeros as padding and only
# flags a row if a nonzero code appears more than once.

# %%
repeat_rows = np.array(has_repeats_per_row(jnp.asarray(recalls)))
n_selected = int(recalls.shape[0])
n_repeat_trials = int(repeat_rows.sum())
pct_repeat = 100.0 * n_repeat_trials / max(1, n_selected)

trials_with_repeats = selected_indices[repeat_rows]

print(f"Trials with any repeated recalls: {n_repeat_trials} ({pct_repeat:.2f}%)")
print(
    "Example trial indices with repeats (in original dataset):",
    preview_list(trials_with_repeats.tolist(), EXAMPLE_HEAD),
)

# %% [markdown]
# ## Count total repeated mentions (beyond first occurrences)

# %%
check_trials_with_repeats, total_extra_mentions = count_repeated_mentions(recalls)

print(
    "Total extra repeated mentions (beyond first occurrences):",
    int(total_extra_mentions),
)
print(
    "Sanity check — repeat trial counts agree:",
    check_trials_with_repeats == n_repeat_trials,
)

# %% [markdown]
# ## Optional: per‑subject summary

# %%
if SHOW_BY_SUBJECT:
    subjects = np.array(data["subject"].flatten())[trial_mask]
    counts = Counter(subjects[repeat_rows].tolist())
    top10 = counts.most_common(10)
    if top10:
        print("Top subjects by repeat‑trial count (subject: count):")
        for sid, cnt in top10:
            print(f"  {sid}: {cnt}")
    else:
        print("No repeated recalls found; per‑subject summary omitted.")

# %% [markdown]
# ## Peek at a few repeat trials (optional)
#
# For quick manual inspection, you can toggle the next cell to print the recall
# rows for the first few indices with repeats.

# %%
SHOW_EXAMPLE_ROWS = False
N_EXAMPLES = 5

if SHOW_EXAMPLE_ROWS and trials_with_repeats.size > 0:
    print("First few repeat trials and their recall rows (0 = padding):")
    for ix in trials_with_repeats[:N_EXAMPLES]:
        row = np.array(data["recalls"][ix])
        print(f"  trial {int(ix)} -> {row.tolist()}")

