"""Check notebook downloads, staged code, example files, and website links."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "guide"))
import prepare
from check_notebooks import CASES


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in (["href"] if tag == "a" else ["src"] if tag in ["img", "script"] else []):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(site):
    for item in CASES:
        name = item["path"]
        raw = (ROOT / name).read_bytes()
        download = Path(name).name if name.startswith("templates/") else Path(name).relative_to("guide")
        assert (site / "downloads" / download).read_bytes() == raw, name
        original = json.loads(raw)
        if name.startswith("templates/"):
            staged = json.loads((ROOT / "guide" / name).read_bytes())
            # Only the first raw front-matter cell gains a presentation link.
            assert staged["cells"][1:] == original["cells"][1:], name
            assert staged["cells"][0]["cell_type"] == original["cells"][0]["cell_type"] == "raw"
            html = (site / Path(name).with_suffix(".html")).read_text(encoding="utf-8")
            url = f"https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/main/{name}"
            assert url in html, f"Missing Colab button: {name}"
    for name in [*prepare._EXAMPLE_FILES, "data/fits/README.md"]:
        assert (site / "templates" / name).read_bytes() == (ROOT / name).read_bytes(), name
    broken = []
    for path in site.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        for link in parser.links:
            if link.startswith("https://colab.research.google.com/github/githubpsyche/jaxcmr/"):
                prefix = "https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/main/"
                assert link.startswith(prefix), f"Unexpected Colab target: {link}"
                assert (ROOT / unquote(link[len(prefix):].split("?")[0])).is_file(), link
            url = urlsplit(link)
            assert url.scheme != "colab", f"Unresolved Colab link: {link}"
            if url.scheme or url.netloc or not url.path:
                continue
            target = site / unquote(url.path.lstrip('/')) if url.path.startswith('/') else path.parent / unquote(url.path)
            if not target.exists():
                broken.append(f"{path.relative_to(site)} -> {link}")
    if broken:
        raise ValueError("Broken local links:\n" + "\n".join(sorted(set(broken))))
    print(f"Verified {len(CASES)} notebook downloads, supplied inputs, Colab targets and local links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    check(parser.parse_args().site)
