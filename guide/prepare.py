"""Stage the Quarto book's notebooks and downloads, with Colab links to main."""

import ast
import json
import os
from pathlib import Path
import re
import shutil

GUIDE = Path(__file__).resolve().parent
ROOT = GUIDE.parent


def registry():
    module = ast.parse((ROOT / "jaxcmr/colab.py").read_text())
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_INPUT_HASHES"
                              for target in node.targets))
    return ast.literal_eval(assignment.value)


def check_destination(path):
    """Reject symlinks before writing or removing generated files."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"Generated output must not use a symlink: {path}")


def stage():
    guide = (GUIDE / "_quarto.yml").read_text()
    names = re.findall(r"^\s*- (templates/\w+\.ipynb)\s*$", guide, re.M)
    if not names:
        raise ValueError("No template chapters found in guide/_quarto.yml.")

    # Read every required source before changing staging or the rendered book.
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
    for name in [*registry(), "data/fits/README.md"]:
        prepared[GUIDE / "templates" / name] = (ROOT / name).read_bytes()

    full_render = os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1"
    output = None
    if full_render:
        output = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
        if not output.is_absolute():
            output = GUIDE / output
        output = Path(os.path.abspath(output))
        check_destination(output)
        # Within the checkout, only docs/ is generated book output. Also reject
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


if __name__ == "__main__":
    stage()
