import contextlib
import json

import numpy as np
import pandas as pd

from jaxcmr.helpers import save_dict_to_hdf5

###############################################################################
#                               1. WORDPOOL & SUBJECTS                        #
###############################################################################


def load_wordpool(wordpool_path):
    """
    Loads a text file of words (one per line), returning a dict:
      word -> 1-based integer ID.
    """
    with open(wordpool_path, "r") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    if not words:
        raise ValueError("Wordpool file is empty.")
    return {w: i for i, w in enumerate(dict.fromkeys(words), start=1)}


def convert_subjects_to_ints(df, subject_col):
    """
    Convert string subject labels to integers (1, 2, 3...).
    Modifies df in-place.
    """
    if subject_col not in df.columns:
        raise ValueError(f"No column {subject_col} in DataFrame.")
    unique_labels = df[subject_col].dropna().unique()
    label_map = {
        label: next_id
        for next_id, label in enumerate(sorted(unique_labels, key=str), start=1)
    }
    df[subject_col] = df[subject_col].apply(lambda x: label_map.get(x, 0)).astype(int)
    return label_map


###############################################################################
#                        2. HELPER FUNCTIONS FOR A BLOCK                      #
###############################################################################


def extract_presentation_items(block_df, wordpool_map):
    """
    Pull out rows type='WORD', sort them, convert studied words -> item IDs in wordpool,
    and produce:
      pres_itemids (list of ints)
      pres_itemnos (list of ints = [1..N])
    """
    pres_df = block_df[block_df["type"] == "WORD"].copy()
    if pres_df.empty:
        return [], []
    sort_col = (
        "serial_position" if "serial_position" in pres_df.columns else "trial_index"
    )
    pres_df = pres_df.sort_values(sort_col, na_position="last")
    studied_words = pres_df["word"].fillna("").str.lower().str.strip().tolist()
    pres_itemids = []
    for w in studied_words:
        if w not in wordpool_map:
            raise ValueError(f"Word '{w}' not found in wordpool.")
        pres_itemids.append(wordpool_map[w])
    pres_itemnos = list(range(1, len(pres_itemids) + 1))
    return pres_itemids, pres_itemnos


def extract_raw_typed(block_df):
    """
    From rows where type='REC_WORD', filter out rows with no data,
    then return:
      typed_words: list of recalled words (lowercased)
      typed_times: list of corresponding RTs (as ints)
    Assumes that the recall rows are already in the correct retrieval order,
    as recorded (e.g., by time_elapsed).
    """
    rec_df = block_df[block_df["type"] == "REC_WORD"].copy()
    # Filter out rows where both 'rec_word' and 'rt' are NaN
    rec_df = rec_df[~(rec_df["rec_word"].isna() & rec_df["rt"].isna())]
    if rec_df.empty:
        return [], []

    # Do not re-sort, since the recall order is already correct (e.g., via time_elapsed)
    rec_df = rec_df.reset_index(drop=True)

    typed_words = rec_df.apply(_extract_typed_word, axis=1).tolist()
    typed_times = rec_df["rt"].fillna(0).astype(float).tolist()
    typed_times = [int(round(t)) for t in typed_times]

    assert len(typed_words) == len(typed_times), (
        f"Mismatch in typed recall extraction: {len(typed_words)} words vs {len(typed_times)} RTs"
    )
    return typed_words, typed_times


def _extract_typed_word(row):
    """
    If rec_word is present, return that (lowercased),
    otherwise parse 'response' which might look like "{'Q0': 'some text'}".
    """
    if w := str(row.get("rec_word", "")).strip().lower():
        return w
    resp = row.get("response", "")
    if isinstance(resp, str) and "Q0" in resp:
        with contextlib.suppress(Exception):
            fixed = resp.replace("'", '"')
            parsed = json.loads(fixed)
            return str(parsed.get("Q0", "")).strip().lower()
    return ""


def build_raw_recall_arrays(typed_words, typed_times, pres_itemids, wordpool_map):
    """
    For each typed word:
      1) rec_itemid = wordpool_map.get(typed_word, -1)
      2) If rec_itemid is in pres_itemids, find its position in that list => correct recall,
         else => intrusion.
      3) Repeated correct recalls are retained with their studied position.
      4) recall_rt is just typed_times.
    Returns (recalls, rec_itemids, recall_rt) with length equal to len(typed_words).
    """
    # Build a mapping from each studied wordpool ID to its within-list position (1-indexed).
    itemid_to_pos = {pid: idx for idx, pid in enumerate(pres_itemids, start=1)}

    recalls = []
    rec_ids = []
    recall_rt = []

    # For each typed word, look up its global wordpool ID, then determine its
    # position in the studied list (if present) or mark as intrusion (-1).
    for w, t in zip(typed_words, typed_times):
        typed_wid = wordpool_map.get(w, -1)
        pos = itemid_to_pos.get(typed_wid, -1)
        recalls.append(pos)
        rec_ids.append(typed_wid)
        recall_rt.append(t)

    return recalls, rec_ids, recall_rt


def build_clean_recall_arrays(recalls, recall_rt):
    """
    Produce clean recall arrays by filtering out intrusions (recall == -1)
    and any repeated correct recalls (duplicate positions) from the raw arrays.
    The clean_recall_rt simply contains the corresponding raw RTs for the first occurrence
    of each correct recall.
    """
    clean_recalls = []
    clean_rts = []
    seen = set()
    for pos, rt in zip(recalls, recall_rt):
        if pos == -1:
            continue
        if pos in seen:
            continue
        seen.add(pos)
        clean_recalls.append(pos)
        clean_rts.append(rt)
    return clean_recalls, clean_rts


def parse_single_block(block_df, wordpool_map, subject_id, session_id, list_id):
    """
    For one (subject, session, list) block:
      1) Extract presentation items.
      2) Extract typed recall words and times.
      3) Build raw recall arrays.
      4) Build clean recall arrays by simply filtering out intrusions.
      5) Return a dict of the processed fields.
    """
    if block_df.empty:
        return {
            "subject": subject_id,
            "session": session_id,
            "list_number": list_id,
            "listLength": 0,
            "pres_itemids": [],
            "pres_itemnos": [],
            "recalls": [],
            "rec_itemids": [],
            "recall_rt": [],
            "clean_recalls": [],
            "clean_recall_rt": [],
        }
    pres_itemids, pres_itemnos = extract_presentation_items(block_df, wordpool_map)
    typed_words, typed_times = extract_raw_typed(block_df)
    recalls, rec_itemids, recall_rt = build_raw_recall_arrays(
        typed_words, typed_times, pres_itemids, wordpool_map
    )
    clean_recs, clean_rts = build_clean_recall_arrays(recalls, recall_rt)
    return {
        "subject": subject_id,
        "session": session_id,
        "list_number": list_id,
        "listLength": len(pres_itemids),
        "pres_itemids": pres_itemids,
        "pres_itemnos": pres_itemnos,
        "recalls": recalls,
        "rec_itemids": rec_itemids,
        "recall_rt": recall_rt,
        "clean_recalls": clean_recs,
        "clean_recall_rt": clean_rts,
    }


###############################################################################
#                      3. PROCESS THE ENTIRE CSV                               #
###############################################################################


def parse_jspsych_csv(df, wordpool_path, subject_col, session_col, list_col):
    # 1) Copy the DataFrame to avoid modifying the original.
    df2 = df.copy()

    # 2) Load the wordpool mapping from the specified file.
    wordpool_map = load_wordpool(wordpool_path)

    # 3) Convert subject labels (strings) to integer codes.
    convert_subjects_to_ints(df2, subject_col)

    # 4) Ensure session and list columns exist and are numeric.
    # If session_col doesn't exist, set it to 1 for all rows.
    if session_col not in df2.columns:
        df2[session_col] = 1
    else:
        df2[session_col] = df2[session_col].fillna(1)
    df2[session_col] = df2[session_col].astype(int)

    # Similarly, if list_col doesn't exist, set it to 0 for all rows.
    df2[list_col] = 0 if list_col not in df2.columns else df2[list_col].fillna(0)
    df2[list_col] = df2[list_col].astype(int)

    # 5) Group by (subject, session, list) and process each block.
    group_cols = [subject_col, session_col, list_col]
    block_dicts = []
    for (subj_val, sess_val, l_val), block_df in df2.groupby(group_cols):
        # Process each group/block.
        # Convert the raw list value to 1-based indexing for list_id.
        block_result = parse_single_block(
            block_df=block_df,
            wordpool_map=wordpool_map,
            subject_id=int(subj_val),
            session_id=int(sess_val),
            list_id=(l_val + 1),  # 1-based indexing
        )
        block_dicts.append(block_result)

    # 6) Aggregate the block results and pad to 2D numpy arrays.
    # Here we choose to use the "clean" recall arrays.
    aggregated = {
        "subject": [],
        "session": [],
        "list_number": [],
        "listLength": [],
        "pres_itemids": [],
        "pres_itemnos": [],
        "recalls": [],
        # "rec_itemids": [],
        "irt": [],
        # "clean_recalls": [],
        # "clean_irt": [],
    }
    for b in block_dicts:
        aggregated["subject"].append(b["subject"])
        aggregated["session"].append(b["session"])
        aggregated["list_number"].append(b["list_number"])
        aggregated["listLength"].append(b["listLength"])
        aggregated["pres_itemids"].append(b["pres_itemids"])
        aggregated["pres_itemnos"].append(b["pres_itemnos"])

        # use clean arrays for modeling
        # aggregated["recalls"].append(b["recalls"])
        # aggregated["rec_itemids"].append(b["rec_itemids"])
        # aggregated["recall_rt"].append(b["recall_rt"])
        # aggregated["clean_recalls"].append(b["clean_recalls"])
        # aggregated["clean_recall_rt"].append(b["clean_recall_rt"])
        aggregated["recalls"].append(b["clean_recalls"])
        aggregated["irt"].append(b["clean_recall_rt"])

    return pad_fields(aggregated)


###############################################################################
#                   4. ZERO-PADDING FIELDS TO NUMPY ARRAYS                    #
###############################################################################


def pad_fields(aggregated_dict):
    """
    Convert each field (list-of-lists or list-of-scalars) into a 2D numpy array.
      - Scalars become arrays of shape (N,1).
      - Lists are padded with zeros to the maximum length.
    """
    final = {}
    n = len(aggregated_dict["subject"])  # number of blocks
    for key, val_list in aggregated_dict.items():
        if any(isinstance(x, list) for x in val_list):
            max_len = max(len(x) for x in val_list)
            padded = []
            for x in val_list:
                row = x if isinstance(x, list) else [x]
                row += [0] * (max_len - len(row))
                padded.append(row)
            final[key] = np.array(padded)
        else:
            final[key] = np.array(val_list).reshape(n, 1)
    return final


###############################################################################
#                               5. DRIVER                                     #
###############################################################################

if __name__ == "__main__":
    csv_path = "data/raw/HerremaKahana2024.csv"
    wordpool_path = "data/wordpool_ltpFR3.txt"
    target_data_path = "data/HerremaKahana2024.h5"

    df = pd.read_csv(csv_path)
    embam = parse_jspsych_csv(
        df,
        wordpool_path,
        subject_col="worker_id",
        session_col="session",
        list_col="list",
    )

    for k, v in embam.items():
        print(k, v.shape)

    save_dict_to_hdf5(embam, target_data_path)
