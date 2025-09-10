# %% [markdown]
# # Cohen & Kahana (2022) → `RecallDataset` loader (percent notebook)
#  Used this to help with some steps: https://chatgpt.com/c/68c13121-02dc-8327-9060-bfeb7f3ed0c4?model=gpt-5
# **Goal.** Build a clean, integer‑only dataset for jaxcmr with exactly these required fields:
#
# - `subject : (n_trials, 1)` — int subject code per trial
# - `listLength : (n_trials, 1)` — count of non‑zero presented IDs per trial
# - `pres_itemids : (n_trials, P)` — raw presented global item IDs (padded with 0)
# - `pres_itemnos : (n_trials, P)` — within‑list positions 1..L, padded with 0
# - `rec_itemids : (n_trials, R)` — raw recalled IDs after filtering intrusions (padded with 0)
# - `recalls : (n_trials, R)` — within‑list positions of recalls, mapping **to the first presentation** (padded with 0)
#
# **Also load if available:**
# - `valence : (n_trials, P)` — codes −1/0/+1 aligned with presentations
# - `session : (n_trials, 1)` — 1‑based session index assuming exactly 24 lists/session
#
# **Intrusion filtering rules (per spec):**
# - Drop **negatives** (e.g., −1)
# - Drop **zeros** anywhere in the recall sequence (treat like intrusions)
# - Drop **ELIs** (positive recall IDs not present in that trial’s `pres_itemids`)
# - **Perseverations:** configurable — keep or drop **immediate** repeats (A A). Non‑successive repeats (A B A) are preserved.
#
# **Roadmap**
# 1. Imports & dataset type
# 2. Config toggles & invariants
# 3. `RecallDataset` `TypedDict`
# 4. Small utilities (CSV parsing, subject labels)
# 5. **Preflight raw validation** — alignment checks that *raise* on fatal issues
# 6. **Core loader** — builds padded integer arrays
# 7. **Post‑hoc dataset validation** — sanity checks on the constructed dataset
# 8. Quick‑start usage cell

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Integer
from typing_extensions import NotRequired, TypedDict

from jaxcmr.helpers import save_dict_to_hdf5
from jaxcmr.typing import RecallDataset

# %% [markdown]
# ## 1) Config toggles & invariants
# - `KEEP_SUCCESSIVE_PERSEVERATIONS`: if `False`, drops **only** immediate repeats (A A → keep first A, drop second).
# - `SESSION_LISTS`: hard assumption: every valid subject has `n_lists % 24 == 0`.

# %%
# If False, filter only successive repeats (A A -> keep first A, drop next A).
# Non-successive repeats (A B A) are preserved.
KEEP_SUCCESSIVE_PERSEVERATIONS: bool = True

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

        # Missing files
        for p in (pres_path, rec_path, eval_path):
            if not p.exists():
                missing_files += 1
                fatal_issues.append(f"Missing file for {label}: {p}")
        if missing_files:
            continue

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
    *,
    keep_successive_perseverations: bool | None = None,
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
    if keep_successive_perseverations is None:
        keep_successive_perseverations = KEEP_SUCCESSIVE_PERSEVERATIONS

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

            # Filter recalls: drop negatives, zeros, ELIs; handle successive perseverations
            filtered_ids: List[int] = []
            prev_id: int | None = None
            pset = set(first_pos.keys())
            for rv in r_row:
                if rv <= 0:
                    continue  # drop negatives and zeros
                if rv not in pset:
                    continue  # ELI intrusion
                if (
                    not keep_successive_perseverations
                    and prev_id is not None
                    and rv == prev_id
                ):
                    continue  # drop immediate repeat
                filtered_ids.append(rv)
                prev_id = rv

            mapped_pos = [first_pos[rv] for rv in filtered_ids]

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
                continue
            if rid == 0 and rpos != 0:
                first_ok = False
                break
            if rid != 0 and rpos == 0:
                first_ok = False
                break
            if rid != 0:
                if first_pos.get(rid, None) != int(rpos):
                    first_ok = False
                    break
        if not first_ok:
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
    target_data_path = "data/CohenKahana2022.h5"
    preflight = validate_raw_cohen_kahana_2022(raw)  # raises on fatal issues
    ds = load_cohen_kahana_2022(raw, keep_successive_perseverations=True)
    report = validate_RecallDataset(ds)
    save_dict_to_hdf5(ds, target_data_path)
    print(report)

# %%
