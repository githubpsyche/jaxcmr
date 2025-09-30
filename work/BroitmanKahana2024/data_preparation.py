# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import os
import glob
import numpy as np
import pandas as pd
import h5py
from typing import List, Dict, Any

# === Configuration toggles ===
BIDS_ROOT = "."                   # root of your BIDS dataset
OUTPUT_FILE = "combined_subjects.h5"
FILTER_6_SESSION_SUBJECTS = False  # only include subjects with ≥ 6 beh sessions
DROP_INTRUSIONS = True            # skip recalls of items not in study list
DROP_REPEATED_RECALLS = True      # skip repeated recalls within trial

# === Helper functions ===

def process_beh_file(
    tsv_path: str,
    subject_idx: int,
    session_idx: int
) -> List[Dict[str, Any]]:
    """
    Read one BIDS-beh.tsv and return per-trial dicts:
      {
        'subject': int,
        'session': int,
        'studied_items': List[str],
        'recalled_items': List[str]
      }
    """
    df = pd.read_csv(tsv_path, sep='\t')
    starts = df.index[df['trial_type'] == 'TRIAL_START'].tolist()
    if not starts:
        raise ValueError(f"No TRIAL_START in {tsv_path}")
    
    trials = []
    for i, st in enumerate(starts):
        ed = starts[i+1] if i+1 < len(starts) else len(df)
        tdf = df.iloc[st:ed]
        # 1) STUDY list
        study = tdf[tdf['trial_type']=="WORD"].sort_values('serialpos')
        if study.empty:
            continue
        studied_items = study['item_name'].astype(str).tolist()
        # 2) RAW recall
        rec = tdf[tdf['trial_type']=="REC_WORD"]
        raw_recalled = rec['item_name'].astype(str).tolist()
        # 3) Filter intrusions & repeats
        seen = set()
        recalled_items = []
        for itm in raw_recalled:
            if DROP_INTRUSIONS and itm not in studied_items:
                continue
            if DROP_REPEATED_RECALLS and itm in seen:
                continue
            seen.add(itm)
            recalled_items.append(itm)
        trials.append({
            'subject':       subject_idx,
            'session':       session_idx,
            'studied_items': studied_items,
            'recalled_items':recalled_items
        })
    return trials

def encode_pres_itemnos(studied_items: List[str]) -> List[int]:
    """1-based within-list indices, reuse on repeats."""
    first_idx = {}
    out = []
    ctr = 0
    for itm in studied_items:
        if itm not in first_idx:
            ctr += 1
            first_idx[itm] = ctr
        out.append(first_idx[itm])
    return out

def encode_recalls(
    studied_items: List[str],
    recalled_items: List[str]
) -> List[int]:
    """Map recalls to first study-position (1-based)."""
    out = []
    for itm in recalled_items:
        try:
            pos = studied_items.index(itm) + 1
            out.append(pos)
        except ValueError:
            # intrusion not in list, skip
            continue
    return out

# === 1) Discover all beh.tsv paths ===
pattern = os.path.join(BIDS_ROOT, "sub-*", "ses-*", "beh", "*_beh.tsv")
all_paths = sorted(glob.glob(pattern))

# === 2) Group by subject & session ===
subjects = {}
for p in all_paths:
    parts = p.split(os.sep)
    subj = parts[1].replace("sub-", "")
    ses  = int(parts[2].replace("ses-", ""))
    subjects.setdefault(subj, []).append((ses, p))

# === 3) Optionally filter to subjects with ≥6 sessions ===
if FILTER_6_SESSION_SUBJECTS:
    subjects = {s: svr for s, svr in subjects.items() if len(svr) >= 6}

# === 4) Process every file ===
all_trials: List[Dict[str,Any]] = []
for sub_i, (subj_label, sess_list) in enumerate(subjects.items(), start=1):
    for ses_i, path in sorted(sess_list, key=lambda x: x[0]):
        all_trials.extend(process_beh_file(path, sub_i, ses_i))

# === 5) Determine max list & recall lengths ===
max_study   = max(len(t['studied_items']) for t in all_trials)
max_recall  = max(len(t['recalled_items']) for t in all_trials)
N_trials    = len(all_trials)

# === 6) Allocate arrays ===
subject_arr    = np.zeros((N_trials, 1), dtype=int)
session_arr    = np.zeros((N_trials, 1), dtype=int)
listLength_arr = np.zeros((N_trials, 1), dtype=int)
pres_itemnos   = np.zeros((N_trials, max_study), dtype=int)
recalls_arr    = np.zeros((N_trials, max_recall), dtype=int)

# === 7) Fill arrays ===
for i, trial in enumerate(all_trials):
    study   = trial['studied_items']
    recall  = trial['recalled_items']
    subject_arr   [i, 0] = trial['subject']
    session_arr   [i, 0] = trial['session']
    listLength_arr[i, 0] = len(study)
    pres_itemnos  [i, :len(study)]   = encode_pres_itemnos(study)
    recalls_arr   [i, :len(recall)]  = encode_recalls(study, recall)

# === 8) Save to HDF5 ===
with h5py.File(OUTPUT_FILE, 'w') as f:
    for name, data in {
        "subject": subject_arr,
        "session": session_arr,
        "listLength": listLength_arr,
        "pres_itemnos": pres_itemnos,
        "recalls": recalls_arr
    }.items():
        f.create_dataset(name, data=data, compression="gzip")

print(f"Saved EMBAM dataset with {N_trials} trials to '{OUTPUT_FILE}'")
for name, arr in [
    ("subject", subject_arr),
    ("session", session_arr),
    ("listLength", listLength_arr),
    ("pres_itemnos", pres_itemnos),
    ("recalls", recalls_arr)
]:
    print(f"  • {name}: {arr.shape}")
