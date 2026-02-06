"""Prepare a recall-style dataset from VRT raw Excel exports.

This script:
1) Extracts written reports ("utterances") from the VRT `.xlsx` exports.
2) Assigns each utterance to one of 11 film clips using sentence-transformers.
3) Loads participant tags (film condition, intervention, intentionality order).
3) Exports integer-coded event streams for film cues ("blurred reminders") and foils.
4) Writes an audit CSV and an HDF5 file compatible with `jaxcmr.helpers.load_data`.

The exported HDF5 contains only integer arrays (no floating-point values, no NaNs).
Time/timestamp fields are intentionally excluded from the HDF5 output.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from jaxcmr.helpers import save_dict_to_hdf5

# Small "domain language" type aliases used throughout the script.
Film = Literal["emotional", "neutral"]
Task = Literal["intrusion", "free_recall"]


# These dataclasses are just lightweight containers for structured data.
# They let us give names to fields (instead of juggling many parallel lists),
# which makes the later data-wrangling steps easier to understand and debug.
@dataclass(frozen=True)
class VrtRow:
    """One row from a VRT sheet export."""

    row_index: int
    a: int | None
    b: int | None
    c: str | None
    d: int | None
    f: int | None
    h: str | None


@dataclass(frozen=True)
class VrtUtterance:
    """A written report extracted from a VRT export."""

    subject: int
    task: Task
    row_index: int
    digit: int | None
    background_code: str | None
    background_type: int | None
    cue_code: str | None
    cue_clip_number: int | None
    cue_row_index: int | None
    text: str


@dataclass(frozen=True)
class VrtClip:
    """One of the 11 clips for a given film condition."""

    film: Film
    clip_number: int
    title: str
    description: str


@dataclass(frozen=True)
class VrtReferenceSentence:
    """A reference sentence used for semantic matching."""

    film: Film
    clip_number: int
    sentence: str


@dataclass(frozen=True)
class VrtUtteranceMatch:
    """A clip assignment for a single utterance."""

    subject: int
    task: Task
    row_index: int
    digit: int | None
    background_code: str | None
    background_type: int | None
    cue_code: str | None
    cue_clip_number: int | None
    cue_row_index: int | None
    text: str
    film: Film | None
    predicted_clip_number: int | None
    predicted_similarity: float | None
    predicted_margin: float | None
    predicted_reference: str | None
    recall_clip_number: int | None


@dataclass(frozen=True)
class VrtPidTags:
    """Participant-level tags loaded from the PID metadata sheet."""

    subject: int
    emotion: int
    intervention: int
    intentionality: int


def _xlsx_cell_to_value(cell: Any, shared_strings: list[str]) -> str | None:
    """Return the decoded value for a spreadsheet cell element."""
    # Local import keeps this script dependency-light (no pandas/openpyxl needed).
    import xml.etree.ElementTree as ET

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    # In the worksheet XML, the text content is often stored inside a `<v>` tag.
    value = cell.find("m:v", ns)
    if value is None or value.text is None:
        return None
    # Excel stores many strings in a separate "shared string table" and then
    # references them by index. If the cell says `t="s"`, the `<v>` is an index.
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    # Otherwise, `<v>` typically contains a literal number/text representation.
    return value.text


def read_xlsx_cell_rows(
    path: Path,
    worksheet_path: str = "xl/worksheets/sheet1.xml",
) -> list[tuple[int, dict[str, str | None]]]:
    """Return sheet rows as column-letter-to-value mappings.

    Args:
      path: Path to an `.xlsx` file.
      worksheet_path: Workbook-internal worksheet XML path.

    Returns:
      List of (row_index, row_values) pairs. `row_index` matches Excel's 1-based row
      numbering, and `row_values` is keyed by Excel column letters.
    """
    import xml.etree.ElementTree as ET

    # A `.xlsx` file is literally a ZIP archive of XML files.
    with zipfile.ZipFile(path) as workbook:
        # `sharedStrings.xml` contains the shared string table used by many cells.
        shared_strings_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared_strings = [
            t.text or ""
            for t in shared_strings_root.iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
            )
        ]

        # The worksheet itself is another XML file with `<row>` and `<c>` (cell) tags.
        sheet_root = ET.fromstring(workbook.read(worksheet_path))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows: list[tuple[int, dict[str, str | None]]] = []
        for row in sheet_root.findall("m:sheetData/m:row", ns):
            # Excel row numbers are 1-based in the XML.
            row_index = int(row.attrib["r"])
            values: dict[str, str | None] = {}
            for cell in row.findall("m:c", ns):
                # Cell references look like "A1", "C14", etc. We keep only the column letters.
                ref = cell.attrib.get("r", "")
                col_match = re.match(r"[A-Z]+", ref)
                if col_match is None:
                    continue
                col = col_match.group(0)
                values[col] = _xlsx_cell_to_value(cell, shared_strings)
            rows.append((row_index, values))
    return rows


def read_vrt_xlsx_rows(path: Path) -> list[VrtRow]:
    """Return parsed sheet rows from a VRT `.xlsx` export.

    Args:
      path: Path to a VRT `.xlsx` file.

    Returns:
      List of rows in sheet order with columns A–H decoded.
    """
    # First read the raw cell table as strings keyed by Excel column letters.
    cell_rows = read_xlsx_cell_rows(path)

    # Helper: treat empty cells as missing, otherwise parse to int.
    def to_int(v: str | None) -> int | None:
        return int(v) if v is not None and v != "" else None

    rows: list[VrtRow] = []
    for row_index, values in cell_rows:
        # Map only the columns we care about into a typed record.
        rows.append(
            VrtRow(
                row_index=row_index,
                a=to_int(values.get("A")),
                b=to_int(values.get("B")),
                c=values.get("C"),
                d=to_int(values.get("D")),
                f=to_int(values.get("F")),
                h=values.get("H"),
            )
        )
    return rows


def parse_subject_and_task(path: Path) -> tuple[int, Task]:
    """Return (subject_id, task) parsed from a VRT file name."""
    # VRT files are expected to look like: `p123_..._Intrusion_....xlsx`
    # or `p123_..._FreeRecall_....xlsx`.
    match = re.match(r"^p(?P<subject>\d+)_", path.name)
    if match is None:
        raise ValueError(f"Could not parse subject from filename: {path.name}")
    subject = int(match.group("subject"))

    if "_Intrusion_" in path.name:
        return subject, "intrusion"
    if "_FreeRecall_" in path.name:
        return subject, "free_recall"
    raise ValueError(f"Could not parse task from filename: {path.name}")


def load_pid_tags(path: Path) -> dict[int, VrtPidTags]:
    """Return participant tags keyed by subject id.

    Args:
      path: Path to the PID tags `.xlsx` file.

    Returns:
      Mapping from subject id to tag values.
    """
    # This workbook is our "source of truth" for participant metadata like:
    # - which film condition they saw (emotion),
    # - whether they got the intervention,
    # - and their intentionality-order code.
    rows = read_xlsx_cell_rows(path)
    if not rows:
        raise ValueError(f"No rows found in PID tags workbook: {path}")

    # The first row is treated as the header. We look up columns by header name.
    header_row_index, header_values = rows[0]
    header_by_col: dict[str, str] = {}
    for col, value in header_values.items():
        if value is None:
            continue
        header_by_col[col] = value.strip().lower()

    def find_column(name: str) -> str:
        for col, header in header_by_col.items():
            if header == name:
                return col
        raise ValueError(
            f"PID tags workbook missing required header {name!r} in row {header_row_index}."
        )

    pid_col = find_column("pid")
    emotion_col = find_column("emotion")
    intervention_col = find_column("intervention")
    intentionality_col = find_column("intentionality")

    def require_int(value: str | None, field: str, row_index: int) -> int:
        # Some spreadsheets store ints as "1.0". Converting via float handles that.
        if value is None or value.strip() == "":
            raise ValueError(f"Missing {field} in PID tags row {row_index}.")
        return int(float(value))

    tags: dict[int, VrtPidTags] = {}
    for row_index, values in rows[1:]:
        # Skip completely empty rows.
        pid_raw = values.get(pid_col)
        if pid_raw is None or pid_raw.strip() == "":
            continue
        # Participant IDs are numeric; again we parse via float for safety.
        subject = int(float(pid_raw))
        if subject in tags:
            raise ValueError(f"Duplicate subject {subject} in PID tags workbook.")
        tags[subject] = VrtPidTags(
            subject=subject,
            emotion=require_int(values.get(emotion_col), "emotion", row_index),
            intervention=require_int(values.get(intervention_col), "intervention", row_index),
            intentionality=require_int(values.get(intentionality_col), "intentionality", row_index),
        )
    return tags


def film_from_emotion_code(emotion: int) -> Film:
    """Return film condition implied by the PID tags emotion code."""
    # In this project, `emotion` is the authoritative code for which film variant
    # a participant saw. We map it into a readable string label.
    if emotion == 1:
        return "emotional"
    if emotion == 0:
        return "neutral"
    raise ValueError(f"Unexpected emotion code {emotion!r}; expected 0 or 1.")


def parse_coded_clip_number(code: str | None) -> int | None:
    """Return a clip number parsed from a coded cue value, if present.

    The raw exports sometimes include values like "5_1" or "7_IPT" in column C.
    These appear to encode the clip number as the leading integer prefix.

    Args:
      code: Raw code string from column C.
    """
    if code is None:
        return None
    # We only care about the leading integer prefix (e.g., "7" from "7_IPT").
    match = re.match(r"^(?P<num>\d+)", code.strip())
    if match is None:
        return None
    number = int(match.group("num"))
    if 1 <= number <= 11:
        return number
    return None


def parse_scene_number(code: str | None) -> int | None:
    """Return an unrelated scene number parsed from a cue value, if present.

    Args:
      code: Raw code string from column C.
    """
    if code is None:
        return None
    # Foils in the raw data are named like "scene1", "scene2", ..., "scene68".
    match = re.match(r"^scene(?P<num>\d+)$", code.strip().lower())
    if match is None:
        return None
    number = int(match.group("num"))
    if 1 <= number <= 68:
        return number
    return None


def extract_vrt_utterances(
    rows: Sequence[VrtRow],
    subject: int,
    task: Task,
    max_rows_since_clip_code: int | None = None,
) -> list[VrtUtterance]:
    """Return written reports (utterances) extracted from VRT rows.

    Args:
      rows: Parsed VRT rows.
      subject: Subject identifier.
      task: Task name ("intrusion" or "free_recall").
      max_rows_since_clip_code: Maximum row distance allowed between a film cue and
        the next utterance for them to be paired. When None, no limit is applied.

    Returns:
      List of utterances in temporal order.
    """
    # We scan the full event stream and pull out only the rows where a participant
    # typed something (column H). Those are our "utterances".
    #
    # While we scan, we also keep track of the most recent film cue (D == 1) so
    # we can attach cue metadata to the next utterance.
    utterances: list[VrtUtterance] = []
    pending_cue_clip_number: int | None = None
    pending_cue_row_index: int | None = None
    pending_cue_code: str | None = None

    for row in rows:
        # Film cues ("blurred reminders") show up as D == 1 with a coded value in C
        # like "7_1" or "7_IPT". We treat that as "cue clip number = 7".
        row_cue_clip_number = parse_coded_clip_number(row.c) if row.d == 1 else None
        if row_cue_clip_number is not None:
            # If multiple cues occur before an utterance, we keep the most recent
            # one, because it is the one most plausibly related to the next report.
            pending_cue_clip_number = row_cue_clip_number
            pending_cue_row_index = row.row_index
            pending_cue_code = row.c

        # Column H contains free text. Only rows with non-empty text are "utterances".
        text = (row.h or "").strip()
        if not text:
            continue

        # Default: no cue attached. We only fill these if there is a pending cue.
        cue_clip_number: int | None = None
        cue_code: str | None = None
        cue_row_index: int | None = None

        if pending_cue_clip_number is not None and pending_cue_row_index is not None:
            # Optional safeguard: only pair a cue to an utterance if they are close
            # in the row stream. If `max_rows_since_clip_code` is None, we allow any
            # distance (the cue is still consumed after a single utterance).
            within_row_window = (
                max_rows_since_clip_code is None
                or row.row_index - pending_cue_row_index <= max_rows_since_clip_code
            )
            if within_row_window:
                cue_clip_number = pending_cue_clip_number
                cue_code = pending_cue_code
                cue_row_index = pending_cue_row_index

        # Create one structured record per utterance so later steps can be written
        # as "data transforms" rather than a lot of fragile index bookkeeping.
        utterances.append(
            VrtUtterance(
                subject=subject,
                task=task,
                row_index=row.row_index,
                digit=row.b,
                background_code=row.c,
                background_type=row.d,
                cue_code=cue_code,
                cue_clip_number=cue_clip_number,
                cue_row_index=cue_row_index,
                text=text,
            )
        )

        # Consume the pending cue after the first utterance that follows it. This
        # ensures a single cue event is never paired to multiple utterances.
        pending_cue_clip_number = None
        pending_cue_row_index = None
        pending_cue_code = None
    return utterances


_WORD_TO_CLIP_NUMBER: dict[str, int] = {
    # The scoring markdown numbers clips with words ("Clip One", "Clip Two", ...).
    # We map those words to integers so we can store clip numbers as ints.
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
}


def parse_vrt_clips(scoring_md_path: Path) -> list[VrtClip]:
    """Parse emotional and neutral clip descriptions from the VRT scoring document.

    Args:
      scoring_md_path: Path to `VRT_Data_Scoring_Info.md`.

    Returns:
      List of clips for both films. Expected length is 22 (11 emotional + 11 neutral).
    """
    # The scoring document is a markdown file with sections like:
    #   # Film Content: Emotional
    #   ### **Clip One: <title> [..]**
    # We parse those headers and then collect the descriptive lines under them.
    lines = scoring_md_path.read_text(encoding="utf-8").splitlines()

    film: Film | None = None
    clips: list[VrtClip] = []
    current_number: int | None = None
    current_title: str | None = None
    buffer: list[str] = []

    header_re = re.compile(r"^### \*\*Clip (?P<num_word>[^:]+): (?P<title>.*?) \[")

    def flush_current() -> None:
        # Helper that "finalizes" the clip we've been collecting so far and adds it
        # to the output list. (We call this when we encounter a new clip header.)
        nonlocal buffer, current_number, current_title
        if film is None or current_number is None or current_title is None:
            buffer = []
            current_number = None
            current_title = None
            return
        description = " ".join(line.strip() for line in buffer).strip()
        clips.append(
            VrtClip(
                film=film,
                clip_number=current_number,
                title=current_title.strip(),
                description=description,
            )
        )
        buffer = []
        current_number = None
        current_title = None

    for line in lines:
        stripped = line.strip()
        if stripped == "# Film Content: Emotional":
            # Switch to the emotional film section.
            flush_current()
            film = "emotional"
            continue
        if stripped == "# Film Content: Neutral":
            # Switch to the neutral film section.
            flush_current()
            film = "neutral"
            continue

        header_match = header_re.match(stripped)
        if header_match is not None and film is not None:
            # New clip header found: finalize the previous clip and start a new one.
            flush_current()
            num_word = header_match.group("num_word").strip().lower()
            if num_word not in _WORD_TO_CLIP_NUMBER:
                raise ValueError(
                    f"Unrecognized clip number word {num_word!r} in: {stripped}"
                )
            current_number = _WORD_TO_CLIP_NUMBER[num_word]
            current_title = header_match.group("title").strip()
            continue

        if film is not None and current_number is not None:
            # Lines under a clip header are treated as free-form description text.
            if stripped:
                buffer.append(stripped)

    flush_current()
    return clips


def split_into_sentences(text: str) -> list[str]:
    """Return a simple sentence split for matching short utterances."""
    # This is intentionally simple: we normalize whitespace and then split at
    # punctuation. (This is not a full NLP sentence tokenizer.)
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+", normalized)


def build_reference_sentences(clips: Sequence[VrtClip]) -> list[VrtReferenceSentence]:
    """Return per-clip reference sentences derived from the scoring descriptions.

    Args:
      clips: Parsed clip descriptions.

    Returns:
      A flattened list of sentences labeled by film and clip number.
    """
    # We build a small "reference library" of sentences that describe each clip.
    # Later we embed both the references and each participant utterance, and then
    # assign an utterance to the clip with the most similar reference sentence.
    references: list[VrtReferenceSentence] = []
    for clip in clips:
        # Use both the title and each sentence from the description as potential
        # targets for matching.
        sentences = [clip.title, *split_into_sentences(clip.description)]
        for sentence in sentences:
            cleaned = sentence.strip()
            # Skip very short fragments that tend to be uninformative in embeddings.
            if len(cleaned) < 20:
                continue
            references.append(
                VrtReferenceSentence(
                    film=clip.film, clip_number=clip.clip_number, sentence=cleaned
                )
            )
    return references


def load_sentence_transformer(model_name: str, device: str | None) -> Any:
    """Return a loaded SentenceTransformer model.

    Args:
      model_name: SentenceTransformer model name or path.
      device: Optional torch device override (e.g., "cpu", "cuda").
    """
    # We import sentence-transformers lazily so that simply importing this module
    # doesn't require ML dependencies. The dependency is only needed when you
    # actually run semantic matching.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for semantic clip matching. "
            "Install the dev dependencies (e.g., `uv sync --dev` or "
            "`pip install -e '.[dev]'`)."
        ) from exc

    return SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)


def encode_texts(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Return L2-normalized embeddings for the provided texts.

    Args:
      model: Loaded SentenceTransformer model.
      texts: Texts to embed.
      batch_size: Batch size for encoding.

    Returns:
      A 2D float array of shape (len(texts), embedding_dim).
    """
    # `model.encode` returns one vector per input string. We request numpy output
    # and normalization so that dot-products correspond to cosine similarity.
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def parse_cue_suffix(code: str | None) -> str | None:
    """Return the suffix component of a film cue code.

    The VRT raw exports encode film cue images in column C using values like
    `7_1` or `7_IPT`, where the prefix is a 1–11 clip number and the suffix
    distinguishes the film condition.

    Args:
      code: Raw code string from column C.
    """
    if code is None:
        return None
    # Split values like "7_IPT" into:
    # - prefix: clip number (7)
    # - suffix: a tag like "IPT" or "1"
    match = re.match(r"^\s*(?P<num>\d+)_(?P<suffix>.+?)\s*$", code)
    if match is None:
        return None
    number = int(match.group("num"))
    if not (1 <= number <= 11):
        return None
    suffix = match.group("suffix").strip()
    return suffix or None


def collect_cue_suffixes(rows: Sequence[VrtRow]) -> set[str]:
    """Return all cue-code suffixes observed in film cue rows."""
    # This was an earlier approach to infer film condition from the cue-code
    # suffix pattern. We keep it around for diagnostics, but the PID tags sheet
    # is now the authoritative source of film condition.
    suffixes: set[str] = set()
    for row in rows:
        if row.d != 1:
            continue
        suffix = parse_cue_suffix(row.c)
        if suffix is not None:
            suffixes.add(suffix)
    return suffixes


def infer_film_from_suffixes(suffixes: set[str]) -> Film:
    """Return the film condition implied by cue-code suffixes.

    The neutral film uses a single `_1` suffix for all cue codes, while the
    emotional film uses IPT and/or varied suffixes.

    Args:
      suffixes: Suffix values observed for a subject or trial.
    """
    # Neutral film: every cue ends with "_1" (so the suffix set is {"1"}).
    # Emotional film: suffixes vary (including "IPT"), so the set is larger.
    if not suffixes:
        raise ValueError("No cue-code suffixes found; cannot infer film condition.")
    return "neutral" if suffixes == {"1"} else "emotional"


def infer_film_by_subject(
    rows_by_trial: Mapping[tuple[int, Task], Sequence[VrtRow]],
) -> tuple[dict[int, Film], dict[int, set[str]]]:
    """Return film condition and cue suffixes per subject.

    Args:
      rows_by_trial: Raw rows for each (subject, task) trial.

    Returns:
      (film_by_subject, suffixes_by_subject):
        - film_by_subject: Mapping from subject id to inferred film condition.
        - suffixes_by_subject: Mapping from subject id to the union of observed suffixes.
    """
    # This function is useful as a sanity-check for the raw data: if a subject
    # shows mixed suffix patterns across tasks, that would be surprising and worth
    # investigating. It is *not* used to set condition in the exported dataset.
    film_by_subject: dict[int, Film] = {}
    suffixes_by_subject: dict[int, set[str]] = {}

    film_by_trial: dict[tuple[int, Task], Film] = {}
    suffixes_by_trial: dict[tuple[int, Task], set[str]] = {}
    for key, rows in rows_by_trial.items():
        suffixes = collect_cue_suffixes(rows)
        suffixes_by_trial[key] = suffixes
        film_by_trial[key] = infer_film_from_suffixes(suffixes)

    for (subject, _task), film in film_by_trial.items():
        existing = film_by_subject.get(subject)
        if existing is None:
            film_by_subject[subject] = film
        elif existing != film:
            raise ValueError(
                f"Subject {subject} has inconsistent film inference across tasks: "
                f"{existing!r} vs {film!r}."
            )

    for (subject, _task), suffixes in suffixes_by_trial.items():
        suffixes_by_subject.setdefault(subject, set()).update(suffixes)

    return film_by_subject, suffixes_by_subject


def _clip_scores_for_film(
    utterance_similarities: np.ndarray,
    reference_films: np.ndarray,
    reference_clip_numbers: np.ndarray,
    film: Film,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-clip scores and best reference indices for a single utterance.

    Args:
      utterance_similarities: Similarities to each reference sentence (shape [n_ref]).
      reference_films: Film labels for reference sentences (shape [n_ref]).
      reference_clip_numbers: Clip numbers for reference sentences (shape [n_ref]).
      film: Film to score against.

    Returns:
      (clip_scores, clip_best_ref_index):
        - clip_scores: shape [11], max similarity for each clip.
        - clip_best_ref_index: shape [11], index of the best matching reference sentence.
    """
    # `utterance_similarities` contains one similarity score per reference sentence.
    # We want a single score per clip, so we take the maximum score among that clip's
    # reference sentences.
    clip_scores = np.full(11, -np.inf, dtype=np.float32)
    clip_best_ref_index = np.full(11, -1, dtype=int)
    for clip_number in range(1, 12):
        mask = (reference_films == film) & (reference_clip_numbers == clip_number)
        if not np.any(mask):
            continue
        masked_indices = np.flatnonzero(mask)
        masked_scores = utterance_similarities[masked_indices]
        best_local = int(masked_scores.argmax())
        best_ref_index = int(masked_indices[best_local])
        clip_scores[clip_number - 1] = float(masked_scores[best_local])
        clip_best_ref_index[clip_number - 1] = best_ref_index
    return clip_scores, clip_best_ref_index


def match_utterances_to_clips(
    utterances: Sequence[VrtUtterance],
    reference_sentences: Sequence[VrtReferenceSentence],
    film_by_subject: Mapping[int, Film],
    model_name: str,
    batch_size: int,
    device: str | None,
    min_similarity: float | None,
    min_margin: float | None,
) -> list[VrtUtteranceMatch]:
    """Return clip assignments for utterances via semantic matching.

    Args:
      utterances: Utterances to match.
      reference_sentences: Reference sentences derived from the scoring document.
      film_by_subject: Mapping from subject id to film condition.
      model_name: SentenceTransformer model name or path.
      batch_size: Batch size for encoding.
      device: Optional torch device override (e.g., "cpu", "cuda").
      min_similarity: Optional minimum similarity required to keep an embedding match.
      min_margin: Optional minimum margin (best - second best) required to keep a match.

    Returns:
      List of per-utterance matches.
    """
    # Build arrays for the reference library (the "known" clip descriptions).
    reference_texts = [r.sentence for r in reference_sentences]
    reference_films = np.asarray([r.film for r in reference_sentences], dtype=object)
    reference_clip_numbers = np.asarray(
        [r.clip_number for r in reference_sentences], dtype=int
    )

    # The "unknowns" we want to classify are the participant utterances.
    utterance_texts = [u.text for u in utterances]

    # Embed references and utterances in the same vector space, then compute
    # cosine similarity via dot product (because embeddings are normalized).
    model = load_sentence_transformer(model_name=model_name, device=device)
    reference_embeddings = encode_texts(model, reference_texts, batch_size=batch_size)
    utterance_embeddings = encode_texts(model, utterance_texts, batch_size=batch_size)
    sim = utterance_embeddings @ reference_embeddings.T

    matches: list[VrtUtteranceMatch] = []
    for i, utterance in enumerate(utterances):
        # The PID tags tell us which film condition this subject saw, so we only
        # score an utterance against references from that film.
        film: Film | None = film_by_subject.get(utterance.subject)

        predicted_clip: int | None = None
        predicted_similarity: float | None = None
        predicted_margin: float | None = None
        predicted_reference: str | None = None

        if film is not None:
            # Step 1: compute a best score for each of the 11 clips.
            clip_scores, clip_best_ref_index = _clip_scores_for_film(
                utterance_similarities=sim[i],
                reference_films=reference_films,
                reference_clip_numbers=reference_clip_numbers,
                film=film,
            )
            # Step 2: pick the best clip, and record its similarity score.
            best_clip_index = int(clip_scores.argmax())
            predicted_clip = best_clip_index + 1
            predicted_similarity = float(clip_scores[best_clip_index])

            # Margin is a simple confidence measure: how much better was the best
            # clip than the runner-up?
            sorted_scores = np.sort(clip_scores)
            if sorted_scores.size >= 2:
                predicted_margin = float(sorted_scores[-1] - sorted_scores[-2])

            # Record which reference sentence was the strongest match (useful for auditing).
            best_ref_index = int(clip_best_ref_index[best_clip_index])
            if 0 <= best_ref_index < len(reference_texts):
                predicted_reference = reference_texts[best_ref_index]

        def keep_embedding_match() -> bool:
            # Optional quality thresholds. When a threshold is provided, we drop
            # low-confidence matches rather than forcing a clip assignment.
            if predicted_clip is None:
                return False
            if min_similarity is not None and (
                predicted_similarity if predicted_similarity is not None else -np.inf
            ) < min_similarity:
                return False
            if min_margin is not None and (
                predicted_margin if predicted_margin is not None else -np.inf
            ) < min_margin:
                return False
            return True

        # IMPORTANT: semantic matching is the only supported way to code recalls.
        # If a match doesn't pass thresholds, we mark it as unmatched (None) and
        # downstream code will drop it from the recall sequences.
        recall_clip = predicted_clip if keep_embedding_match() else None

        # We keep both:
        # - the raw predicted fields (for auditing),
        # - and the final `recall_clip_number` (used in the exported dataset).
        matches.append(
            VrtUtteranceMatch(
                subject=utterance.subject,
                task=utterance.task,
                row_index=utterance.row_index,
                digit=utterance.digit,
                background_code=utterance.background_code,
                background_type=utterance.background_type,
                cue_code=utterance.cue_code,
                cue_clip_number=utterance.cue_clip_number,
                cue_row_index=utterance.cue_row_index,
                text=utterance.text,
                film=film,
                predicted_clip_number=predicted_clip,
                predicted_similarity=predicted_similarity,
                predicted_margin=predicted_margin,
                predicted_reference=predicted_reference,
                recall_clip_number=recall_clip,
            )
        )
    return matches


def keep_first_mentions(
    recalls: Sequence[int],
    cue_clips: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Return (recalls, cue_clips) with repeats removed.

    Repeats are defined by recalled clip number; later mentions of a clip are
    removed while keeping the first mention (and its aligned cue clip).

    Args:
      recalls: Recalled clip numbers.
      cue_clips: Cue clip numbers aligned with `recalls`.

    Returns:
      (recalls_unique, cue_clips_unique): Sequences with duplicates removed.
    """
    if len(recalls) != len(cue_clips):
        raise ValueError("recalls and cue_clips must have the same length.")

    # We iterate in order and keep the first time each clip appears.
    seen: set[int] = set()
    recalls_unique: list[int] = []
    cue_unique: list[int] = []
    for recall, cue_clip in zip(recalls, cue_clips):
        if recall in seen:
            continue
        seen.add(recall)
        recalls_unique.append(recall)
        cue_unique.append(cue_clip)
    return recalls_unique, cue_unique


def build_recall_dataset(
    matches: Sequence[VrtUtteranceMatch],
    trial_keys: Iterable[tuple[int, Task]],
    rows_by_trial: Mapping[tuple[int, Task], Sequence[VrtRow]],
    pid_tags_by_subject: Mapping[int, VrtPidTags],
    list_length: int,
) -> dict[str, np.ndarray]:
    """Build a `RecallDataset`-compatible dict for VRT trials.

    Args:
      matches: Per-utterance matches, including semantic `recall_clip_number`.
      trial_keys: Trial identifiers (subject, task) to include.
      rows_by_trial: Raw rows per trial for exporting cue/foil event streams.
      pid_tags_by_subject: Participant tags keyed by subject id.
      list_length: Number of studied items (clips), typically 11.

    Returns:
      Dict of numpy arrays suitable for `save_dict_to_hdf5`.
    """
    # The HDF5 format used by this project expects *2D integer arrays*.
    #
    # Convention used here:
    # - One row per trial, where "trial" means (subject, task).
    # - Fields that are naturally single-valued per trial are shaped (n_trials, 1).
    # - Sequences (like recall order) are padded with zeros on the right so that
    #   every trial has the same number of columns.
    task_code = {"intrusion": 1, "free_recall": 2}
    condition_code = {"emotional": 1, "neutral": 2}

    # Group all utterance matches by trial so we can build per-trial sequences.
    grouped: dict[tuple[int, Task], list[VrtUtteranceMatch]] = {}
    for match in matches:
        grouped.setdefault((match.subject, match.task), []).append(match)

    trial_rows: list[dict[str, object]] = []
    for subject, task in sorted(trial_keys, key=lambda k: (k[0], task_code[k[1]])):
        # Look up the participant metadata (condition, intervention, etc.).
        pid_tags = pid_tags_by_subject.get(subject)
        if pid_tags is None:
            raise ValueError(f"Missing PID tags for subject {subject}.")
        film = film_from_emotion_code(pid_tags.emotion)

        # Sort utterances into the original order they occurred in the sheet.
        ordered_matches = sorted(
            grouped.get((subject, task), []),
            key=lambda m: m.row_index,
        )
        # Drop utterances that we could not confidently match to a clip.
        recall_matches = [m for m in ordered_matches if m.recall_clip_number is not None]

        # `recalls_raw` keeps *all* mentions in order.
        recalls_raw = [int(m.recall_clip_number) for m in recall_matches]

        # `cue_clips_raw` is aligned with `recalls_raw` and stores the cue that
        # immediately preceded the utterance (0 if none).
        cue_clips_raw = [int(m.cue_clip_number or 0) for m in recall_matches]

        # `recalls` removes repeats, keeping only the first mention of each clip.
        recalls, cue_clips = keep_first_mentions(recalls_raw, cue_clips_raw)

        # The full row stream is used to export "event stream" diagnostics and
        # simple counts of reminders/foils shown during the task.
        rows = rows_by_trial.get((subject, task), [])
        reminder_clips = [
            int(clip)
            for row in rows
            if row.d == 1 and (clip := parse_coded_clip_number(row.c)) is not None
        ]
        foil_scenes = [
            int(scene)
            for row in rows
            if row.d == 2 and (scene := parse_scene_number(row.c)) is not None
        ]

        # Build a lookup from Excel row index -> recalled clip number. This lets us
        # tag the event stream with "a recall occurred on this row".
        recall_by_row_index: dict[int, int] = {}
        for match in recall_matches:
            recall_by_row_index.setdefault(match.row_index, int(match.recall_clip_number))

        # Store everything we need to later assemble numpy arrays with padding.
        trial_rows.append(
            {
                "subject": subject,
                "task_code": task_code[task],
                "condition_code": condition_code[film],
                "intervention": pid_tags.intervention,
                "intentionality": pid_tags.intentionality,
                "recalls_raw": recalls_raw,
                "recalls": recalls,
                "cue_clips_raw": cue_clips_raw,
                "cue_clips": cue_clips,
                "rows": rows,
                "reminder_clips": reminder_clips,
                "foil_scenes": foil_scenes,
                "recall_by_row_index": recall_by_row_index,
            }
        )

    if not trial_rows:
        raise ValueError("No trials were constructed from matches.")

    # Figure out how wide each padded array needs to be.
    n_trials = len(trial_rows)
    max_recalls_raw = max(1, max(len(t["recalls_raw"]) for t in trial_rows))  # type: ignore[arg-type]
    max_recalls = max(1, max(len(t["recalls"]) for t in trial_rows))  # type: ignore[arg-type]
    max_events = max(1, max(len(t["rows"]) for t in trial_rows))  # type: ignore[arg-type]
    max_reminders = max(1, max(len(t["reminder_clips"]) for t in trial_rows))  # type: ignore[arg-type]
    max_foils = max(1, max(len(t["foil_scenes"]) for t in trial_rows))  # type: ignore[arg-type]

    # Allocate all output arrays up-front. We initialize with zeros, which doubles
    # as padding (0 means "no value" for variable-length sequences).
    subject_arr = np.zeros((n_trials, 1), dtype=int)
    task_arr = np.zeros((n_trials, 1), dtype=int)
    condition_arr = np.zeros((n_trials, 1), dtype=int)
    intervention_arr = np.zeros((n_trials, 1), dtype=int)
    intentionality_arr = np.zeros((n_trials, 1), dtype=int)
    list_length_arr = np.full((n_trials, 1), list_length, dtype=int)

    # Presentation order is always 1..list_length for each trial.
    pres_itemnos = np.tile(np.arange(1, list_length + 1, dtype=int), (n_trials, 1))
    pres_itemids = np.zeros_like(pres_itemnos)

    recalls_raw_arr = np.zeros((n_trials, max_recalls_raw), dtype=int)
    rec_itemids_raw_arr = np.zeros((n_trials, max_recalls_raw), dtype=int)
    cue_clips_raw_arr = np.zeros((n_trials, max_recalls_raw), dtype=int)

    recalls_arr = np.zeros((n_trials, max_recalls), dtype=int)
    rec_itemids_arr = np.zeros((n_trials, max_recalls), dtype=int)
    cue_clips_arr = np.zeros((n_trials, max_recalls), dtype=int)

    reminder_clips_arr = np.zeros((n_trials, max_reminders), dtype=int)
    foil_scenes_arr = np.zeros((n_trials, max_foils), dtype=int)

    event_row_index_arr = np.zeros((n_trials, max_events), dtype=int)
    event_digit_arr = np.zeros((n_trials, max_events), dtype=int)
    event_background_type_arr = np.zeros((n_trials, max_events), dtype=int)
    event_response_arr = np.zeros((n_trials, max_events), dtype=int)
    event_cue_clip_number_arr = np.zeros((n_trials, max_events), dtype=int)
    event_scene_number_arr = np.zeros((n_trials, max_events), dtype=int)
    event_has_utterance_arr = np.zeros((n_trials, max_events), dtype=int)
    event_recall_clip_number_arr = np.zeros((n_trials, max_events), dtype=int)

    for i, trial in enumerate(trial_rows):
        # --- Trial-level (single-value) metadata ---
        subject = int(trial["subject"])  # type: ignore[arg-type]
        subject_arr[i, 0] = subject
        task_arr[i, 0] = int(trial["task_code"])  # type: ignore[arg-type]
        condition = int(trial["condition_code"])  # type: ignore[arg-type]
        condition_arr[i, 0] = condition
        intervention_arr[i, 0] = int(trial["intervention"])  # type: ignore[arg-type]
        intentionality_arr[i, 0] = int(trial["intentionality"])  # type: ignore[arg-type]

        # --- Item numbering conventions ---
        # pres_itemnos is always 1..11 for each trial.
        # pres_itemids must be unique across film conditions, so we add an offset:
        #   emotional: 1..11
        #   neutral:   12..22
        itemid_offset = list_length * (condition - 1)
        pres_itemids[i, :] = pres_itemnos[i, :] + itemid_offset

        # --- Recall sequences (raw and de-duplicated) ---
        recalls_raw: list[int] = trial["recalls_raw"]  # type: ignore[assignment]
        if recalls_raw:
            recalls_raw_arr[i, : len(recalls_raw)] = np.asarray(recalls_raw, dtype=int)
            rec_itemids_raw_arr[i, : len(recalls_raw)] = (
                np.asarray(recalls_raw, dtype=int) + itemid_offset
            )

        cue_clips_raw: list[int] = trial["cue_clips_raw"]  # type: ignore[assignment]
        if cue_clips_raw:
            cue_clips_raw_arr[i, : len(cue_clips_raw)] = np.asarray(cue_clips_raw, dtype=int)

        recalls: list[int] = trial["recalls"]  # type: ignore[assignment]
        if recalls:
            recalls_arr[i, : len(recalls)] = np.asarray(recalls, dtype=int)
            rec_itemids_arr[i, : len(recalls)] = np.asarray(recalls, dtype=int) + itemid_offset

        cue_clips: list[int] = trial["cue_clips"]  # type: ignore[assignment]
        if cue_clips:
            cue_clips_arr[i, : len(cue_clips)] = np.asarray(cue_clips, dtype=int)

        # --- Simple lists of all reminders/foils shown in the row stream ---
        reminder_clips: list[int] = trial["reminder_clips"]  # type: ignore[assignment]
        if reminder_clips:
            reminder_clips_arr[i, : len(reminder_clips)] = np.asarray(reminder_clips, dtype=int)

        foil_scenes: list[int] = trial["foil_scenes"]  # type: ignore[assignment]
        if foil_scenes:
            foil_scenes_arr[i, : len(foil_scenes)] = np.asarray(foil_scenes, dtype=int)

        # --- Event stream export ---
        # This is a row-by-row representation of the raw sheet, useful for auditing
        # and for analyses that look at how cues relate to subsequent reports.
        rows: Sequence[VrtRow] = trial["rows"]  # type: ignore[assignment]
        recall_by_row_index: dict[int, int] = trial["recall_by_row_index"]  # type: ignore[assignment]
        for j, row in enumerate(rows[:max_events]):
            event_row_index_arr[i, j] = int(row.row_index)
            event_digit_arr[i, j] = int(row.b or 0)
            event_background_type_arr[i, j] = int(row.d or 0)
            event_response_arr[i, j] = int(row.f or 0)
            event_has_utterance_arr[i, j] = 1 if (row.h or "").strip() else 0

            # For every row we compute:
            # - which film cue was shown (if any),
            # - which foil scene was shown (if any),
            # - and which clip was recalled on that row (if any).
            cue_clip = parse_coded_clip_number(row.c) if row.d == 1 else None
            event_cue_clip_number_arr[i, j] = int(cue_clip or 0)
            scene_number = parse_scene_number(row.c) if row.d == 2 else None
            event_scene_number_arr[i, j] = int(scene_number or 0)
            event_recall_clip_number_arr[i, j] = int(recall_by_row_index.get(row.row_index, 0))

    # Bundle all arrays into a dict. `save_dict_to_hdf5` will write each key as a dataset.
    dataset: dict[str, np.ndarray] = {
        "subject": subject_arr,
        "task": task_arr,
        "condition": condition_arr,
        "intervention": intervention_arr,
        "intentionality": intentionality_arr,
        "listLength": list_length_arr,
        "pres_itemnos": pres_itemnos,
        "pres_itemids": pres_itemids,
        "recalls": recalls_arr,
        "rec_itemids": rec_itemids_arr,
        "recalls_raw": recalls_raw_arr,
        "rec_itemids_raw": rec_itemids_raw_arr,
        "cue_clips": cue_clips_arr,
        "cue_clips_raw": cue_clips_raw_arr,
        "reminder_clips": reminder_clips_arr,
        "foil_scenes": foil_scenes_arr,
        "event_row_index": event_row_index_arr,
        "event_digit": event_digit_arr,
        "event_background_type": event_background_type_arr,
        "event_response": event_response_arr,
        "event_cue_clip_number": event_cue_clip_number_arr,
        "event_scene_number": event_scene_number_arr,
        "event_has_utterance": event_has_utterance_arr,
        "event_recall_clip_number": event_recall_clip_number_arr,
    }

    # Final safety checks: enforce "2D integer arrays with non-negative values".
    for key, value in dataset.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"HDF5 payload value for {key!r} is not a numpy array.")
        if value.ndim != 2:
            raise ValueError(f"HDF5 payload value for {key!r} is not 2D.")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"HDF5 payload value for {key!r} is not an integer array.")
        if value.size and value.min() < 0:
            raise ValueError(f"HDF5 payload value for {key!r} contains negative values.")

    return dataset


def write_audit_csv(matches: Sequence[VrtUtteranceMatch], path: Path) -> None:
    """Write a per-utterance audit CSV for clip assignment inspection."""
    # The audit CSV is meant for manual inspection:
    # - what cue preceded each utterance?
    # - what clip did the embedding model predict?
    # - did it pass thresholds and become part of `recalls_raw` / `recalls`?
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "task",
                "row_index",
                "digit",
                "background_type",
                "background_code",
                "cue_code",
                "cue_row_index",
                "cue_clip_number",
                "text",
                "film",
                "predicted_clip_number",
                "predicted_similarity",
                "predicted_margin",
                "predicted_reference",
                "recall_clip_number",
            ],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "subject": match.subject,
                    "task": match.task,
                    "row_index": match.row_index,
                    "digit": match.digit,
                    "background_type": match.background_type,
                    "background_code": match.background_code,
                    "cue_code": match.cue_code,
                    "cue_row_index": match.cue_row_index,
                    "cue_clip_number": match.cue_clip_number,
                    "text": match.text,
                    "film": match.film,
                    "predicted_clip_number": match.predicted_clip_number,
                    "predicted_similarity": match.predicted_similarity,
                    "predicted_margin": match.predicted_margin,
                    "predicted_reference": match.predicted_reference,
                    "recall_clip_number": match.recall_clip_number,
                }
            )


def collect_vrt_utterances(
    input_dir: Path,
    max_subjects: int | None,
    max_rows_since_clip_code: int | None,
) -> tuple[list[VrtUtterance], set[tuple[int, Task]], dict[tuple[int, Task], list[VrtRow]]]:
    """Collect utterances and raw rows from VRT `.xlsx` files in a directory."""
    # Find all VRT exports and keep only one file per (subject, task).
    # If there are multiple exports, we keep the one with the lexicographically
    # largest filename as a simple "latest export" heuristic.
    discovered = sorted(input_dir.glob("p*_*.xlsx"))
    files_by_key: dict[tuple[int, Task], Path] = {}
    for path in discovered:
        key = parse_subject_and_task(path)
        existing = files_by_key.get(key)
        if existing is None or path.name > existing.name:
            files_by_key[key] = path

    # Optionally limit to the first `max_subjects` subjects for quicker iteration.
    subjects = sorted({subject for subject, _ in files_by_key.keys()})
    included_subjects = set(subjects[:max_subjects] if max_subjects is not None else subjects)
    trial_keys = {key for key in files_by_key.keys() if key[0] in included_subjects}

    utterances: list[VrtUtterance] = []
    rows_by_trial: dict[tuple[int, Task], list[VrtRow]] = {}

    session_by_task = {"intrusion": 1, "free_recall": 2}
    for subject, task in sorted(trial_keys, key=lambda k: (k[0], session_by_task[k[1]])):
        # Read the full sheet rows and store them so we can export the event stream.
        path = files_by_key[(subject, task)]
        rows = read_vrt_xlsx_rows(path)
        rows_by_trial[(subject, task)] = rows
        # Extract just the participant's written reports (and attach cue metadata).
        utterances.extend(
            extract_vrt_utterances(
                rows,
                subject=subject,
                task=task,
                max_rows_since_clip_code=max_rows_since_clip_code,
            )
        )
    return utterances, trial_keys, rows_by_trial


def main(argv: Sequence[str] | None = None) -> int:
    """Run VRT extraction, matching, and dataset export."""
    # This CLI is organized as a simple pipeline:
    # 1) Read raw `.xlsx` exports and extract utterances
    # 2) Load participant metadata (PID tags)
    # 3) Parse clip descriptions from the scoring markdown
    # 4) Use sentence-transformers to match utterances to clips
    # 5) Write an audit CSV + a RecallDataset-style HDF5 file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/VRT_data_unprocessed"),
        help="Directory containing VRT raw `.xlsx` files.",
    )
    parser.add_argument(
        "--scoring-md",
        type=Path,
        default=Path("data/raw/VRT_Data_Scoring_Info.md"),
        help="Path to the VRT scoring info markdown (clip descriptions).",
    )
    parser.add_argument(
        "--pid-tags-xlsx",
        type=Path,
        default=Path("data/raw/VRP_pid_tags.xlsx"),
        help="Path to the participant PID tags `.xlsx` workbook.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=Path("data/VRT_clips.h5"),
        help="Output HDF5 path (RecallDataset-compatible).",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/vrt_clip_matching.csv"),
        help="Audit CSV path for per-utterance matches.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name for semantic matching.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Encode batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional minimum similarity to accept an embedding match.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Optional minimum margin (best-second) to accept an embedding match.",
    )
    parser.add_argument(
        "--max-rows-since-clip-code",
        type=int,
        default=None,
        help="Max row distance to pair a film cue to the next text row (default: no limit).",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap on unique subjects (for faster iteration).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # Step 1: read raw exports and extract utterances + raw row streams.
    utterances, trial_keys, rows_by_trial = collect_vrt_utterances(
        args.input_dir,
        max_subjects=args.max_subjects,
        max_rows_since_clip_code=args.max_rows_since_clip_code,
    )
    if not utterances:
        raise ValueError(f"No utterances found under {args.input_dir}.")

    # Step 2: load participant metadata and check that every subject has a row.
    pid_tags_all = load_pid_tags(args.pid_tags_xlsx)
    subjects = sorted({subject for subject, _ in trial_keys})
    missing_tags = [subject for subject in subjects if subject not in pid_tags_all]
    if missing_tags:
        raise ValueError(
            f"Missing PID tags for {len(missing_tags)} subjects, e.g. {missing_tags[:5]}."
        )
    pid_tags_by_subject = {subject: pid_tags_all[subject] for subject in subjects}

    film_by_subject = {
        subject: film_from_emotion_code(pid_tags_by_subject[subject].emotion)
        for subject in subjects
    }
    emotional_count = sum(1 for film in film_by_subject.values() if film == "emotional")
    neutral_count = sum(1 for film in film_by_subject.values() if film == "neutral")
    print(
        "Loaded film condition from PID tags: "
        f"{emotional_count} emotional, {neutral_count} neutral."
    )

    # Step 3: parse the scoring document and turn it into reference sentences.
    clips = parse_vrt_clips(args.scoring_md)
    references = build_reference_sentences(clips)

    # Step 4: semantic matching (the only supported approach) assigns each utterance
    # to a clip number, with optional quality thresholds.
    matches = match_utterances_to_clips(
        utterances=utterances,
        reference_sentences=references,
        film_by_subject=film_by_subject,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        min_similarity=args.min_similarity,
        min_margin=args.min_margin,
    )

    # Step 5: write an audit CSV you can inspect in a spreadsheet.
    write_audit_csv(matches, args.audit_csv)

    # Step 6: build the padded integer arrays and export to HDF5.
    dataset = build_recall_dataset(
        matches=matches,
        trial_keys=trial_keys,
        rows_by_trial=rows_by_trial,
        pid_tags_by_subject=pid_tags_by_subject,
        list_length=11,
    )
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    save_dict_to_hdf5(dataset, str(args.output_h5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
