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

# %% [markdown]
# # Cohen & Kahana (2022) → `RecallDataset` loader (percent notebook)
#
# This script prepares a rectangular, integer-only dataset from the raw
# Cohen & Kahana (2022) files for use in `jaxcmr`.
#
# The **output `RecallDataset`** has one row per trial and the following fields:
#
# - **`subject : (n_trials, 1)`**
#   Integer subject code (e.g., 93 for “LTP093”).
#
# - **`listLength : (n_trials, 1)`**
#   Actual number of non-zero presented items in the list.
#
# - **`pres_itemids : (n_trials, P)`**
#   Global item IDs presented on each trial, padded with zeros to the maximum list length.
#
# - **`pres_itemnos : (n_trials, P)`**
#   Within-list serial positions 1..L for each presented item, padded with zeros.
#
# - **`rec_itemids : (n_trials, R)`**
#   Raw recalled item IDs after filtering. Intrusions may be dropped or retained depending on config flags. Padded with zeros to the maximum recall length.
#
# - **`recalls : (n_trials, R)`**
#   Within-list serial positions of recalled items, padded with zeros.
#   - In-list recalls: positive serial position (1..L).
#   - Extra-list intrusions (ELIs) that are **kept**: `rec_itemids > 0` but `recalls == 0`.
#   - Padding: `rec_itemids == 0` and `recalls == 0`.
#
# - **`valence : (n_trials, P)`**
#   Emotional valence codes (−1, 0, +1) aligned with `pres_itemids`.
#
# - **`session : (n_trials, 1)`**
#   Session index (1-based), assuming exactly 24 lists per session.
#
# ## Conversion issues addressed
#
# - **Intrusions**
#   - **Negatives (−1)** and **zeros** in recall sequences are always dropped.
#   - **ELIs** (positive IDs not in the current list):
#     - If `FILTER_ELIS=True`, they are dropped.
#     - If `FILTER_ELIS=False`, they are preserved in `rec_itemids` and marked with `recalls == 0`.
#     This makes them distinguishable from padding by the joint condition:
#     `rec_itemids > 0 & recalls == 0`.
#
# - **Repeated recalls of the same item**
#   Controlled by `FILTER_REPEATED_RECALLS`.
#   - If `True`, only the first recall of an item is kept.
#   - If `False`, all repeats are retained.
#   This applies to **any** repeats, not just immediate ones.
#
# - **Mapping recalls to positions**
#   Each recalled ID is mapped to the **first** presentation of that ID within the study list.
#   This matches the CMR family’s convention. If the recalled ID was not presented in the list (ELI kept), its position is recorded as 0.
#
# - **Zeros in the middle of recall sequences**
#   These are treated as anomalies; the validator warns but the loader filters them like intrusions.
#

# %%

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax.numpy as jnp
import numpy as np

from jaxcmr.helpers import save_dict_to_hdf5
from jaxcmr.typing import RecallDataset

# %% [markdown]
# ## 1) Config toggles & invariants
# - `KEEP_SUCCESSIVE_PERSEVERATIONS`: if `False`, drops **only** immediate repeats (A A → keep first A, drop second).
# - `SESSION_LISTS`: hard assumption: every valid subject has `n_lists % 24 == 0`.

# %%
# If True, remove any repeated recall of the same item within a trial
# (A ... A) — not just immediate repeats.
FILTER_REPEATED_RECALLS: bool = True

# If True, drop out-of-list intrusions (positive IDs not in the presented set).
FILTER_ELIS: bool = True

# Fatal guard: every subject must have lists multiple of this session size
SESSION_LISTS: int = 24

# %% [markdown]
# ## 3) Validation report dataclasses
# These are simple containers for returning structured information from the validators.


# %%
@dataclass
class RawValidationReport:
    subjects_checked: int
    fatal_issues: List[str]
    warnings: List[str]
    # counters aggregated across subjects
    total_subjects_missing_files: int
    total_row_count_mismatches: int
    total_pres_eval_width_mismatches: int
    total_non_multiple_of_session: int


@dataclass
class DatasetValidationReport:
    n_trials: int
    num_presented_max: int
    num_recalled_max: int
    # filtered counts are not directly reconstructable post-hoc
    filtered_neg_intrusions: int
    filtered_zero_intrusions: int
    filtered_eli_intrusions: int
    filtered_successive_perseverations: int
    # structural checks
    recalls_within_bounds: bool
    mapping_uses_first_presentation: bool


# %% [markdown]
# ## 4) Small utilities (CSV parsing, labels)
# Keep these lean and predictable — no hidden behaviors.


# %%
def _parse_valid_labels(valid_list_path: Path) -> List[str]:
    """Return labels like 'LTP093' from the valid-subjects list.
    Accepts either 'LTP093' or a bare number like '93'."""
    labels: List[str] = []
    for ln in valid_list_path.read_text().splitlines():
        tok = ln.strip()
        if not tok:
            continue
        if tok.upper().startswith("LTP"):
            labels.append(tok.upper())
        else:
            num = int("".join(ch for ch in tok if ch.isdigit()))
            labels.append(f"LTP{num:03d}")
    return labels


def _subject_int_from_label(label: str) -> int:
    """Map 'LTP093' → 93 (int)."""
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits)


def _read_csv_matrix(path: Path) -> List[List[int]]:
    """Parse a comma-separated integer matrix file into a list of int lists.
    Skips blank lines (e.g., trailing blank line)."""
    rows: List[List[int]] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        row = [int(tok.strip()) for tok in ln.split(",") if tok.strip() != ""]
        rows.append(row)
    return rows


# %% [markdown]
# ## 5) Preflight raw validation (raises on fatal)
# Checks across **pres/rec/eval** for each valid subject:
# - Missing files
# - Row count mismatches
# - Per‑trial width mismatch between `pres` and `eval`
# - Lists not a multiple of 24 (session size)
#
# Soft warnings only (do not raise):
# - Zeros in the *middle* of recall sequences
# - Positive recalls not in presentations (ELIs)
# - Negative recalls present
# - Valence codes outside {−1,0,1}


# %%
def validate_raw_cohen_kahana_2022(raw_dir: Path | str) -> RawValidationReport:
    raw_path = Path(raw_dir)
    valid_list_path = raw_path / "valid_subjects_list.txt"
    if not valid_list_path.exists():
        raise FileNotFoundError(f"Missing valid subjects list: {valid_list_path}")

    pres_dir = raw_path / "pres_files"
    rec_dir = raw_path / "rec_files"
    eval_dir = raw_path / "eval_files"

    labels = _parse_valid_labels(valid_list_path)

    fatal_issues: List[str] = []
    warnings: List[str] = []
    missing_files = 0
    row_mismatches = 0
    width_mismatches = 0
    non_multiple = 0

    for label in labels:
        pres_path = pres_dir / f"pres_nos_{label}.txt"
        rec_path = rec_dir / f"rec_nos_{label}.txt"
        eval_path = eval_dir / f"eval_codes_{label}.txt"

        missing_this = False
        for p in (pres_path, rec_path, eval_path):
            if not p.exists():
                missing_files += 1
                missing_this = True
                fatal_issues.append(f"Missing file for {label}: {p}")
        if missing_this:
            continue  # skip this subject only

        pres_rows = _read_csv_matrix(pres_path)
        rec_rows = _read_csv_matrix(rec_path)
        eval_rows = _read_csv_matrix(eval_path)

        # Row counts
        if not (len(pres_rows) == len(rec_rows) == len(eval_rows)):
            row_mismatches += 1
            fatal_issues.append(
                f"Row count mismatch for {label}: pres={len(pres_rows)} rec={len(rec_rows)} eval={len(eval_rows)}"
            )
            continue

        n_lists = len(pres_rows)
        if n_lists % SESSION_LISTS != 0:
            non_multiple += 1
            fatal_issues.append(
                f"Lists not multiple of {SESSION_LISTS} for {label}: n_lists={n_lists}"
            )

        # Per-trial checks
        for i, (p_row, r_row, e_row) in enumerate(zip(pres_rows, rec_rows, eval_rows)):
            # pres <-> eval width
            if len(p_row) != len(e_row):
                width_mismatches += 1
                fatal_issues.append(
                    f"Width mismatch {label} trial {i}: pres width={len(p_row)} vs eval width={len(e_row)}"
                )

            # Soft warnings on recalls
            # zero in the middle (before a later non-zero)
            if any(
                (rv == 0 and any(x != 0 for x in r_row[j + 1 :]))
                for j, rv in enumerate(r_row)
            ):
                warnings.append(f"Zeros in middle of recall seq: {label} trial {i}")

            pset = set(x for x in p_row if x > 0)
            if any((rv > 0 and rv not in pset) for rv in r_row):
                warnings.append(f"ELI intrusions present: {label} trial {i}")
            if any(rv < 0 for rv in r_row):
                warnings.append(f"Negative intrusions present: {label} trial {i}")

            if any(ev not in (-1, 0, 1) for ev in e_row):
                warnings.append(f"Non {-1, 0, 1} valence code: {label} trial {i}")

    if missing_files or row_mismatches or width_mismatches or non_multiple:
        raise ValueError(
            "\n".join(
                [
                    "Fatal raw-data issues detected:",
                    *fatal_issues,
                    "-- end fatal issues --",
                ]
            )
        )

    return RawValidationReport(
        subjects_checked=len(labels),
        fatal_issues=fatal_issues,
        warnings=warnings,
        total_subjects_missing_files=missing_files,
        total_row_count_mismatches=row_mismatches,
        total_pres_eval_width_mismatches=width_mismatches,
        total_non_multiple_of_session=non_multiple,
    )


# %% [markdown]
# ## 6) Core loader
# - Includes **intrusion filtering** and the **first‑presentation** mapping for `recalls`.
# - Computes `listLength` as the count of non‑zero `pres_itemids` per trial.
# - Pads to dataset‑level maxima with zeros.


# %%
def load_cohen_kahana_2022(
    raw_dir: Path | str,
    filter_repeated_recalls: bool | None = None,
    filter_elis: bool | None = None,
) -> RecallDataset:
    """Load the dataset into a rectangular `RecallDataset` (integer arrays only).

    Parameters
    ----------
    raw_dir : Path | str
        Path to `data/raw/CohenKahana2022` with subdirs `pres_files/`, `rec_files/`,
        `eval_files/`, and `valid_subjects_list.txt`.
    keep_successive_perseverations : bool | None
        If given, overrides module-level `KEEP_SUCCESSIVE_PERSEVERATIONS`.

    Returns
    -------
    RecallDataset
        Dict of 2D `jnp.int32` arrays with required and optional fields.
    """
    raw_path = Path(raw_dir)
    if filter_repeated_recalls is None:
        filter_repeated_recalls = FILTER_REPEATED_RECALLS
    if filter_elis is None:
        filter_elis = FILTER_ELIS

    valid_list_path = raw_path / "valid_subjects_list.txt"
    pres_dir = raw_path / "pres_files"
    rec_dir = raw_path / "rec_files"
    eval_dir = raw_path / "eval_files"

    labels = _parse_valid_labels(valid_list_path)

    # First pass: collect variable-length rows; compute maxima
    all_pres: List[List[int]] = []
    all_rec_filtered_ids: List[List[int]] = []
    all_rec_mapped_pos: List[List[int]] = []
    all_valence: List[List[int]] = []

    subject_vec: List[int] = []
    session_vec: List[int] = []
    list_len_vec: List[int] = []

    num_presented_max = 0
    num_recalled_max = 0

    for label in labels:
        pres_path = pres_dir / f"pres_nos_{label}.txt"
        rec_path = rec_dir / f"rec_nos_{label}.txt"
        eval_path = eval_dir / f"eval_codes_{label}.txt"

        pres_rows = _read_csv_matrix(pres_path)
        rec_rows = _read_csv_matrix(rec_path)
        eval_rows = _read_csv_matrix(eval_path)

        # Strong guards (mirror preflight)
        assert len(pres_rows) == len(rec_rows) == len(eval_rows), (
            f"Row count mismatch for {label}: pres={len(pres_rows)} rec={len(rec_rows)} eval={len(eval_rows)}"
        )
        n_lists = len(pres_rows)
        assert n_lists % SESSION_LISTS == 0, (
            f"Lists not multiple of {SESSION_LISTS} for {label}: n_lists={n_lists}"
        )

        subj_int = _subject_int_from_label(label)

        for idx, (p_row, r_row, e_row) in enumerate(
            zip(pres_rows, rec_rows, eval_rows)
        ):
            # width alignment pres <-> eval
            if len(p_row) != len(e_row):
                raise ValueError(
                    f"Width mismatch {label} trial {idx}: pres width={len(p_row)} vs eval width={len(e_row)}"
                )

            # list length = count of non-zero presented IDs
            L = sum(1 for x in p_row if x > 0)
            list_len_vec.append(L)
            session_idx = (idx // SESSION_LISTS) + 1  # 1-based
            session_vec.append(session_idx)
            subject_vec.append(subj_int)

            all_pres.append(p_row)
            all_valence.append(e_row)

            # Map ID -> FIRST presentation position (1-based)
            first_pos: Dict[int, int] = {}
            pos_counter = 0
            for x in p_row:
                if x > 0:
                    pos_counter += 1
                    if x not in first_pos:
                        first_pos[x] = pos_counter

            # Filter recalls: drop negatives/zeros; optionally drop ELIs; optionally drop ANY repeats
            filtered_ids: List[int] = []
            seen_ids: set[int] = set()
            pset = set(first_pos.keys())

            for rv in r_row:
                # negatives and zeros are always dropped
                if rv <= 0:
                    continue

                # out-of-list intrusion?
                if rv not in pset:
                    if filter_elis:
                        continue
                    # else keep the ELI in rec_itemids; it will not map to a within-list position
                    # (we'll handle its 'recalls' value as 0 below)

                # repeated recall?
                if filter_repeated_recalls and rv in seen_ids:
                    continue

                filtered_ids.append(rv)
                seen_ids.add(rv)

            mapped_pos = [first_pos.get(rv, 0) for rv in filtered_ids]

            all_rec_filtered_ids.append(filtered_ids)
            all_rec_mapped_pos.append(mapped_pos)

            num_presented_max = max(num_presented_max, len(p_row))
            num_recalled_max = max(num_recalled_max, len(filtered_ids))

    # Second pass: pad to maxima and assemble arrays
    n_trials = len(all_pres)
    pres_itemids = np.zeros((n_trials, num_presented_max), dtype=np.int32)
    pres_itemnos = np.zeros((n_trials, num_presented_max), dtype=np.int32)
    valence = np.zeros((n_trials, num_presented_max), dtype=np.int32)

    rec_itemids = np.zeros((n_trials, num_recalled_max), dtype=np.int32)
    recalls = np.zeros((n_trials, num_recalled_max), dtype=np.int32)

    for i in range(n_trials):
        prow = all_pres[i]
        vrow = all_valence[i]
        L = sum(1 for x in prow if x > 0)

        # Presented IDs & valence
        pres_itemids[i, : len(prow)] = np.asarray(prow, dtype=np.int32)
        valence[i, : len(vrow)] = np.asarray(vrow, dtype=np.int32)

        # Within-list positions 1..L (0 for padding)
        if L > 0:
            pres_itemnos[i, :L] = np.arange(1, L + 1, dtype=np.int32)

        # Recalls
        rid_row = all_rec_filtered_ids[i]
        rpos_row = all_rec_mapped_pos[i]
        if rid_row:
            rec_itemids[i, : len(rid_row)] = np.asarray(rid_row, dtype=np.int32)
            recalls[i, : len(rpos_row)] = np.asarray(rpos_row, dtype=np.int32)

    subject = jnp.asarray(np.asarray(subject_vec, dtype=np.int32).reshape(n_trials, 1))
    session = jnp.asarray(np.asarray(session_vec, dtype=np.int32).reshape(n_trials, 1))
    listLength = jnp.asarray(
        np.asarray(list_len_vec, dtype=np.int32).reshape(n_trials, 1)
    )

    ds: RecallDataset = {
        "subject": subject,
        "listLength": listLength,
        "pres_itemids": jnp.asarray(pres_itemids),
        "pres_itemnos": jnp.asarray(pres_itemnos),
        "rec_itemids": jnp.asarray(rec_itemids),
        "recalls": jnp.asarray(recalls),
        "valence": jnp.asarray(valence),
        "session": session,
    }
    return ds


# %% [markdown]
# ## 7) Post‑hoc dataset validation
# Lightweight checks on the *constructed* arrays:
# - `recalls` within `[0, listLength]`
# - `rec_itemids[i,j]` maps to the **first** presentation at `recalls[i,j]`
#
# Note: We do **not** attempt to reconstruct exact counts of filtered events; we report −1 as a sentinel.


# %%
def validate_RecallDataset(ds: RecallDataset) -> DatasetValidationReport:
    subject = np.asarray(ds["subject"])  # (n_trials, 1)
    list_len = np.asarray(ds["listLength"]).reshape(-1)
    pres_ids = np.asarray(ds["pres_itemids"])  # (n_trials, P)
    rec_ids = np.asarray(ds["rec_itemids"])  # (n_trials, R)
    recalls = np.asarray(ds["recalls"])  # (n_trials, R)

    n_trials, num_presented_max = pres_ids.shape
    _, num_recalled_max = rec_ids.shape

    # bounds check: recalls in [0, listLength]
    max_per_row = list_len[:, None]
    recalls_within = ((recalls >= 0) & (recalls <= max_per_row)).all()

    # first-presentation mapping check
    first_ok = True
    for i in range(n_trials):
        first_pos: Dict[int, int] = {}
        pos = 0
        for x in pres_ids[i]:
            if x > 0:
                pos += 1
                if x not in first_pos:
                    first_pos[x] = pos
        for rid, rpos in zip(rec_ids[i], recalls[i]):
            if rid == 0 and rpos == 0:
                continue  # padding
            elif rid == 0:
                first_ok = False
                break
            elif rpos == 0:
                # allow only if not an in-list ID (i.e., kept ELI)
                if rid in first_pos:
                    first_ok = False
                    break
            elif first_pos.get(rid) != int(rpos):
                first_ok = False
                break

    # We cannot directly count filtered tokens post-hoc; set sentinels
    filtered_neg = -1
    filtered_zero = -1
    filtered_eli = -1
    filtered_persev = -1

    return DatasetValidationReport(
        n_trials=n_trials,
        num_presented_max=num_presented_max,
        num_recalled_max=num_recalled_max,
        filtered_neg_intrusions=filtered_neg,
        filtered_zero_intrusions=filtered_zero,
        filtered_eli_intrusions=filtered_eli,
        filtered_successive_perseverations=filtered_persev,
        recalls_within_bounds=bool(recalls_within),
        mapping_uses_first_presentation=bool(first_ok),
    )


# %% [markdown]
# ## 8) Quick‑start usage
# This mirrors the intended workflow without entangling validation with loading.

# %%
if __name__ == "__main__":
    raw = Path("data/raw/CohenKahana2022")
    data_tag = "CohenKahana2022"
    preflight = validate_raw_cohen_kahana_2022(raw)  # raises on fatal issues

    for filter_elis in (True, False):
        filter_tag = "noELI" if filter_elis else "withELI"
        ds = load_cohen_kahana_2022(
            raw,
            filter_repeated_recalls=FILTER_REPEATED_RECALLS,
            filter_elis=filter_elis,
        )
        report = validate_RecallDataset(ds)
        save_dict_to_hdf5(ds, f"data/{data_tag}_{filter_tag}.h5")
        print(report)

# %%
