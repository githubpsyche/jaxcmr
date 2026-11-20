"""Check website navigation, themes, links, and downloadable content."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "guide"))
from jaxcmr.examples import _EXAMPLE_FILES
from check_notebooks import CASES
from prepare import notebook_code_options


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.contents = set()
        self.sidebar = []
        self.themes = {}
        self.theme_switch = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "button" and "quarto-color-scheme-toggle" in classes:
            self.theme_switch = True
        if tag == "link" and attrs.get("id") == "quarto-bootstrap":
            self.themes[attrs["data-mode"]] = attrs["href"]
        if tag == "a" and "sidebar-link" in classes and "href" in attrs:
            self.sidebar.append(attrs["href"])
        if tag == "a" and "data-scroll-target" in attrs:
            self.contents.add(unquote(urlsplit(attrs["href"]).fragment))
        for key in (["href"] if tag in ["a", "link"] else ["src"] if tag in ["img", "script"] else []):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(site):
    notebook_code_options()
    reference_sidebar = None
    site_themes = None
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
    for name in [*_EXAMPLE_FILES, "data/fits/README.md"]:
        assert (site / "templates" / name).read_bytes() == (ROOT / name).read_bytes(), name
    broken = []
    for path in site.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        assert parser.theme_switch and parser.themes.keys() == {"light", "dark"}, f"Missing theme controls: {path}"
        themes = {mode: (path.parent / href).resolve() for mode, href in parser.themes.items()}
        if site_themes is None:
            site_themes = themes
        assert themes == site_themes, f"Inconsistent theme stylesheets: {path}"
        missing = parser.contents - parser.ids
        assert not missing, f"Missing contents targets in {path}: {sorted(missing)}"
        if path.parent == site / "reference":
            if reference_sidebar is None:
                reference_sidebar = parser.sidebar
            assert parser.sidebar and parser.sidebar == reference_sidebar, f"Inconsistent API sidebar: {path}"
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
    print(f"Verified {len(CASES)} notebook downloads, supplied inputs, local links, contents targets, themes and API navigation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    check(parser.parse_args().site)
