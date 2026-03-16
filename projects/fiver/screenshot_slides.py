"""Take headless Chrome screenshots of each revealjs slide."""

import subprocess
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML = Path(__file__).parent / "index.html"
OUTDIR = Path(__file__).parent / "screenshots"
N_SLIDES = 10  # title + 9 content slides (0-indexed in revealjs)
WIDTH = 1920
HEIGHT = 1080


def screenshot_slide(slide_index: int, output_path: Path) -> None:
    """Capture a single slide via headless Chrome.

    Parameters
    ----------
    slide_index : int
        Zero-based slide index for revealjs hash navigation.
    output_path : Path
        Where to save the PNG screenshot.
    """
    url = f"file://{HTML.resolve()}#/{slide_index}"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={output_path}",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            url,
        ],
        capture_output=True,
        timeout=30,
    )


def main() -> None:
    """Screenshot all slides."""
    OUTDIR.mkdir(exist_ok=True)
    for i in range(N_SLIDES):
        out = OUTDIR / f"slide_{i}.png"
        screenshot_slide(i, out)
        print(f"Slide {i}: {out}")


if __name__ == "__main__":
    main()
