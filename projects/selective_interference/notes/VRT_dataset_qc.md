# VRT dataset QC snapshot

This snapshot was computed from the current `data/VRT_clips.h5` file on disk.

## Trial and subject counts

- Trials: **480**
- Subjects: **240**

## Trial-level distributions

- `task`:
  - `1` (intrusion): **240**
  - `2` (free_recall): **240**
- `condition`:
  - `1` (emotional): **240**
  - `2` (neutral): **240**
- `intervention`:
  - `0`: **240**
  - `1`: **240**
- `intentionality`:
  - `1`: **240**
  - `2`: **240**
- `listLength`: **11** (unique)

## Recall counts (nonzero per trial)

- `recalls` (unique-first):
  - mean: **8.16**
  - median: **9**
- `recalls_raw` (all mentions):
  - mean: **17.73**
  - median: **16**

## Cue prevalence (aligned to recall events)

- `cue_clips` rate among nonzero `recalls`: **0.717**
- `cue_clips_raw` rate among nonzero `recalls_raw`: **0.710**

## Cue/foil stream counts per trial

- `reminder_clips` (film cues) per trial:
  - mean: **22**
  - median: **22**
- `foil_scenes` per trial:
  - mean: **68**
  - median: **68**

## Event stream length

- `event_row_index` rows per trial:
  - mean: **270**
  - median: **270**

