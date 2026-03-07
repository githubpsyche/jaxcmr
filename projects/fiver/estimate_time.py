"""Estimate speaking time from revealjs speaker notes in a .qmd file."""

import re
import sys
from pathlib import Path

WPM = 130  # deliberate presentation pace

def extract_notes(qmd_text: str) -> list[dict[str, str]]:
    """Extract slide titles and speaker notes from a .qmd string.

    Parameters
    ----------
    qmd_text : str
        Raw contents of a .qmd file.

    Returns
    -------
    list[dict[str, str]]
        One dict per slide with keys 'title' and 'notes'.
    """
    slides = []

    # check for title-slide data-notes in YAML frontmatter
    frontmatter_match = re.match(r"\A---\s*\n(.*?)\n---", qmd_text, re.DOTALL)
    if frontmatter_match:
        yaml_block = frontmatter_match.group(1)
        data_notes_match = re.search(
            r'data-notes:\s*["\']?(.*?)["\']?\s*$', yaml_block, re.MULTILINE
        )
        if data_notes_match:
            title_match = re.search(r"title:\s*\"(.*?)\"", yaml_block)
            title = title_match.group(1) if title_match else "Title Slide"
            slides.append({"title": title, "notes": data_notes_match.group(1)})

    # strip YAML frontmatter
    body = re.sub(r"\A---.*?---\s*", "", qmd_text, count=1, flags=re.DOTALL)

    # split on slide boundaries (## headings)
    slides_raw = re.split(r"(?=^## )", body, flags=re.MULTILINE)
    slides_raw = [s for s in slides_raw if s.strip()]

    for raw in slides_raw:
        # extract title
        title_match = re.match(r"^##\s*(.*?)(?:\s*\{.*\})?\s*$", raw, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "(untitled)"

        # extract notes block
        notes_match = re.search(
            r":::\s*\{\.notes\}\s*\n(.*?)\n\s*:::", raw, re.DOTALL
        )
        notes = notes_match.group(1).strip() if notes_match else ""
        slides.append({"title": title, "notes": notes})

    return slides


def word_count(text: str) -> int:
    """Count words in a string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of whitespace-delimited tokens.
    """
    return len(text.split())


def format_time(seconds: float) -> str:
    """Format seconds as M:SS.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like '1:23'.
    """
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main(path: str) -> None:
    """Print per-slide and total timing estimates.

    Parameters
    ----------
    path : str
        Path to the .qmd file.
    """
    qmd_text = Path(path).read_text()
    slides = extract_notes(qmd_text)

    total_words = 0
    print(f"{'Slide':<50} {'Words':>6} {'Time':>6}")
    print("-" * 64)

    for i, slide in enumerate(slides, 1):
        wc = word_count(slide["notes"])
        total_words += wc
        seconds = (wc / WPM) * 60
        label = f"{i}. {slide['title'][:45]}"
        print(f"{label:<50} {wc:>6} {format_time(seconds):>6}")

    total_seconds = (total_words / WPM) * 60
    print("-" * 64)
    print(f"{'TOTAL':<50} {total_words:>6} {format_time(total_seconds):>6}")
    print(f"\nAssuming {WPM} WPM (deliberate presentation pace)")
    budget = 5 * 60
    delta = budget - total_seconds
    if delta >= 0:
        print(f"Buffer: {format_time(delta)} under 5:00")
    else:
        print(f"OVER by {format_time(-delta)} — consider cutting ~{int(-delta * WPM / 60)} words")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "index.qmd"
    main(target)
