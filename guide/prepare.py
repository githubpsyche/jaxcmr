"""Prepare API references and stage notebook pages and downloads for Quarto."""

import json
import os
from pathlib import Path
import re
import shutil
import sys

GUIDE = Path(__file__).resolve().parent
ROOT = GUIDE.parent


# Resolve source modules and the example catalog from this checkout.
sys.path.insert(0, str(ROOT))


def check_destination(path):
    """Reject symlinks before writing or removing generated files."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"Generated output must not use a symlink: {path}")


def notebook_code_options(*, update=False):
    """Check or refresh the portable code settings embedded in published notebooks."""
    from textwrap import indent

    import yaml

    project = (GUIDE / "_quarto.yml").read_text()
    defaults = yaml.safe_load(project)["format"]["html"]
    commands = yaml.safe_load((GUIDE / "_commands.yml").read_text())["format"]["html"]
    names = re.findall(r"^\s*- ((?:templates|overview|tutorials)/[\w-]+\.ipynb)\s*$", project, re.M)
    for name in ["developer-setup.ipynb", *names]:
        path = (ROOT if name.startswith("templates/") else GUIDE) / name
        notebook = json.loads(path.read_bytes())
        cell = notebook["cells"][0]
        front = "".join(cell["source"])
        html = yaml.safe_load(front.split("---", 2)[1])["format"]["html"]
        expected = commands if name == "developer-setup.ipynb" else defaults
        # Preserve each notebook's standalone options and its content-specific links.
        for key in ("code-fold", "code-tools", "code-line-numbers", "code-overflow"):
            if key not in html or html[key] == expected[key]:
                continue
            if not update:
                raise ValueError(f"{path}: refresh {key} with guide/prepare.py --sync-notebook-format")
            block = yaml.safe_dump({key: expected[key]}, sort_keys=False)
            front, replaced = re.subn(
                rf"^    {key}:[^\n]*\n(?:^      [^\n]*\n)*",
                lambda _: indent(block, "    "), front, flags=re.M,
            )
            if replaced != 1:
                raise ValueError(f"Could not update {key} in {path}")
        if front != "".join(cell["source"]):
            cell["source"] = front.splitlines(keepends=True)
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def stage():
    from jaxcmr.examples import _EXAMPLE_FILES

    guide = (GUIDE / "_quarto.yml").read_text(encoding="utf-8")
    names = re.findall(r"^\s*- (templates/\w+\.ipynb)\s*$", guide, re.M)
    if not names:
        raise ValueError("No template pages found in guide/_quarto.yml.")

    # Read every required source before changing staging or the rendered site.
    prepared = {}
    downloads = GUIDE / "downloads"
    for name in names:
        raw = (ROOT / name).read_bytes()
        prepared[downloads / Path(name).name] = raw
        notebook = json.loads(raw)
        front = notebook["cells"][0]
        source = "".join(front["source"])
        link = ("      - text: Open in Colab\n"
                "        icon: box-arrow-up-right\n"
                f"        href: https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/main/{name}\n"
                "        target: _blank\n        rel: noopener\n")
        source = source.rstrip().removesuffix("---") + link + "---\n"
        front["source"] = source.splitlines(keepends=True)
        prepared[GUIDE / name] = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()

    for name in ["developer-setup.ipynb", *re.findall(r"^\s*- ((?:overview|tutorials)/[\w-]+\.ipynb)\s*$", guide, re.M)]:
        prepared[downloads / name] = (GUIDE / name).read_bytes()
    for source in (GUIDE / "examples").glob("*.ipynb"):
        prepared[downloads / "examples" / source.name] = source.read_bytes()
    for name in _EXAMPLE_FILES:
        raw = (ROOT / name).read_bytes()
        prepared[GUIDE / "templates" / name] = raw
    prepared[GUIDE / "templates/data/fits/README.md"] = (ROOT / "data/fits/README.md").read_bytes()

    full_render = os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1"
    output = None
    if full_render:
        output = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
        if not output.is_absolute():
            output = GUIDE / output
        output = Path(os.path.abspath(output))
        check_destination(output)
        # Within the checkout, only docs/ is generated site output. Also reject
        # ancestors of the checkout when an external output directory is used.
        if ROOT.is_relative_to(output) or (
            output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "docs")
        ):
            raise ValueError(f"Refusing to clear a source directory: {output}")

    staging = [GUIDE / "templates", downloads]
    for folder in staging:
        check_destination(folder)
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Generated staging must not contain a symlink: {path}")
    for destination in prepared:
        check_destination(destination)

    for destination, content in prepared.items():
        if destination.is_file() and destination.read_bytes() == content:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    if full_render:
        for folder in staging:
            for path in folder.rglob("*"):
                if path.is_file() and path not in prepared:
                    path.unlink()
            for path in sorted(folder.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
        if output.exists():
            shutil.rmtree(output)

    print(f"Staged {len(names)} templates with Colab links to main.")


def prepare_reference_navigation() -> None:
    """Use the API sidebar's groups and links for the homepage and reference index."""
    import yaml

    project = yaml.safe_load((GUIDE / "_quarto.yml").read_text())
    sidebar = next(bar for bar in project["website"]["sidebar"] if bar.get("id") == "api-reference")
    overview = GUIDE / "reference/index.qmd"
    front_matter = overview.read_text().split("---", 2)[1]
    summaries = yaml.safe_load(front_matter)["page-summaries"]

    home = [f"- [{sidebar['title']}]({overview.relative_to(GUIDE).as_posix()})"]
    index = []
    for group in sidebar["contents"]:
        if "section" not in group:
            continue
        title, link = group["section"], group["href"]
        home.append(f"  - [{title}]({link})")
        page, _, anchor = link.partition("#")
        if anchor:
            index.extend([f"## {title} {{#{anchor}}}", "", "| Area | Contents |", "|---|---|"])
            for entry in group["contents"]:
                filename = Path(entry["href"]).name
                label = entry.get("section") or entry["text"]
                index.append(f"| [{label}]({filename}) | {summaries[filename]} |")
        else:
            filename = Path(page).name
            index.extend([f"## [{title}]({filename})", "", summaries[filename]])
        index.append("")

    output = overview.parent / "_generated"
    output.mkdir(exist_ok=True)
    (output / "_homepage-api.qmd").write_text("\n".join(home) + "\n")
    (output / "_reference-sections.qmd").write_text("\n".join(index))


def prepare_references(inputs: set[str] | None = None) -> None:
    """Build the requested module pages from their Quartodoc group definitions."""
    import yaml
    from griffe import Attribute, TypeAlias
    from quartodoc import Builder, get_object
    from quartodoc.layout import DocAttribute

    for page in sorted((GUIDE / "reference").glob("*.qmd")):
        if inputs is not None and page.relative_to(GUIDE).as_posix() not in inputs:
            continue
        configuration = page.with_name(f"_{page.stem}.yml")
        if not configuration.exists():
            continue
        specification = yaml.safe_load(configuration.read_text())["quartodoc"]
        builder = Builder.from_quarto_config({"quartodoc": {
            **specification,
            "style": "single-page",
            "source_dir": str(ROOT),
            "dir": str(GUIDE / "reference/_generated" / page.stem),
            "parser": "auto",
            "renderer": {
                "style": "_api_renderer.py",
                "module": specification["package"],
                "header_level": 2,
                "show_signature_annotations": False,
                "display_name": "full",
                "table_style": "table",
            },
        }})
        # Quartodoc 0.11 predates Griffe's Python 3.12 TypeAlias objects.
        # Adapt only explicit alias entries, retaining their source and docstrings.
        module = get_object(specification["package"], parser="auto")
        for section in builder.layout.sections[0].contents:
            for index, entry in enumerate(section.contents):
                obj = module[entry.name]
                if isinstance(obj, TypeAlias):
                    attribute = Attribute(
                        obj.name, value=obj.value, docstring=obj.docstring,
                        parent=obj.parent, lineno=obj.lineno, endlineno=obj.endlineno,
                    )
                    attribute.labels.add("type-alias")
                    section.contents[index] = DocAttribute(
                        name=entry.name, obj=attribute, anchor=obj.path,
                    )
        # Name the single generated page as an include, rather than a site page.
        builder.layout.sections[0].path = "_body"
        builder.out_inventory = str(Path(builder.dir) / "objects.json")
        builder.build()
        print(f"Prepared API reference: {page.relative_to(GUIDE)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--sync-notebook-format"]:
        notebook_code_options(update=True)
        sys.exit()
    os.chdir(GUIDE)
    inputs = set(os.environ.get("QUARTO_PROJECT_INPUT_FILES", "").splitlines())
    full_render = os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1"
    if full_render or not inputs or inputs & {"index.ipynb", "reference/index.qmd"}:
        prepare_reference_navigation()
    prepare_references(None if full_render or not inputs else inputs)
    # A reference-only render has no notebook or example-download inputs.
    if full_render or not inputs or any(
        not name.startswith("reference/") for name in inputs
    ):
        stage()
