# VRT_clips.h5 field guide

All datasets are **2D integer arrays** with **0 used for padding**. No floats or NaNs.

## Trial-level fields (shape: `[n_trials, 1]`)

- `subject`: Subject identifier.
- `task`: Task code.
  - `1` = intrusion
  - `2` = free_recall
- `condition`: Film condition code.
  - `1` = emotional
  - `2` = neutral
- `intervention`: Intervention code from the PID tags spreadsheet (integer; mapping is
  defined in the source sheet).
- `intentionality`: Intentionality-order code from the PID tags spreadsheet (integer;
  mapping is defined in the source sheet).
- `listLength`: List length (always `11` for VRT clips).

## Presentation fields (shape: `[n_trials, 11]`)

- `pres_itemnos`: Within-list study positions (always `1..11` per trial).
- `pres_itemids`: Global item IDs unique across films.
  - Emotional film: `1..11`
  - Neutral film: `12..22`

## Recall sequences

All recall arrays are aligned by **recall event index** (columns are time-ordered recall events).

- `recalls` (shape: `[n_trials, max_recalls]`): **Unique-first** recall sequence using
  within-list positions `1..11`. Repeats removed after first mention.
- `rec_itemids` (shape: `[n_trials, max_recalls]`): Global item IDs corresponding to `recalls`.

- `recalls_raw` (shape: `[n_trials, max_recalls_raw]`): **All mentions** recall sequence
  using within-list positions `1..11`. Repeats preserved.
- `rec_itemids_raw` (shape: `[n_trials, max_recalls_raw]`): Global item IDs corresponding
  to `recalls_raw`.

## Cue alignment (per recall event)

Cue arrays are aligned to the corresponding recall arrays:

- `cue_clips` (shape: `[n_trials, max_recalls]`): Clip number (`1..11`) of the **cue shown
  immediately before** the recall in `recalls`. `0` if no cue preceded that recall.
- `cue_clips_raw` (shape: `[n_trials, max_recalls_raw]`): Same as `cue_clips`, aligned to
  `recalls_raw`.

The cue pairing rule is **one-to-one**: a cue can only be assigned to the **next**
utterance that follows it (and is then consumed).
ok 
## Summary lists of cues and foils (per trial)

These are **not aligned to recall events**; they are just lists in row order:

- `reminder_clips` (shape: `[n_trials, max_reminders]`): All film cue clip numbers shown
  during the trial (`1..11`).
- `foil_scenes` (shape: `[n_trials, max_foils]`): All foil scene numbers shown during the
  trial (`1..68`).

## Event stream (per raw row)

These fields align to the **raw row stream** from the VRT Excel export. Each trial
has its own row stream; all are padded with zeros to a common width.

- `event_row_index`: Excel row number.
- `event_digit`: Digit shown on screen (column B; `1..9`, `0` if missing).
- `event_background_type`: Background type (column D).
  - `1` = film cue image
  - `2` = unrelated scene (foil)
  - `3` = black
- `event_response`: Button response (column F).
  - `1` = “1” pressed
  - `2` = no press (withheld)
  - `3` = space bar (memory response)
- `event_cue_clip_number`: Parsed cue clip number from column C when `event_background_type == 1`.
- `event_scene_number`: Parsed foil scene number from column C when `event_background_type == 2`.
- `event_has_utterance`: `1` if column H contains any text on that row, else `0`.
- `event_recall_clip_number`: Matched clip number for a recall occurring on that row
  (based on semantic matching); `0` if no matched recall on that row.

