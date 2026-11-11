"""Stage the Quarto book's notebooks and downloads, with Colab links to main."""

import ast
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent


def registry():
    module = ast.parse((ROOT / "jaxcmr/colab.py").read_text())
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_INPUT_HASHES"
                              for target in node.targets))
    return ast.literal_eval(assignment.value)


def stage():
    guide = (ROOT / "guide/_quarto.yml").read_text()
    names = re.findall(r"^\s*- (templates/\w+\.ipynb)\s*$", guide, re.M)
    if not names:
        raise ValueError("No template chapters found in guide/_quarto.yml.")
    for folder in ["guide/templates", "guide/downloads", "docs"]:
        path = ROOT / folder
        if path.is_symlink():
            raise ValueError(f"Generated output must not be a symlink: {folder}")
        if path.exists():
            shutil.rmtree(path)
    downloads = ROOT / "guide/downloads"
    downloads.mkdir(parents=True)
    for name in names:
        raw = (ROOT / name).read_bytes()
        (downloads / Path(name).name).write_bytes(raw)
        notebook = json.loads(raw)
        front = notebook["cells"][0]
        source = "".join(front["source"])
        link = ("      - text: Open in Colab\n"
                "        icon: box-arrow-up-right\n"
                f"        href: https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/main/{name}\n"
                "        target: _blank\n        rel: noopener\n")
        source = source.rstrip().removesuffix("---") + link + "---\n"
        front["source"] = source.splitlines(keepends=True)
        destination = ROOT / "guide" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")

    for name in ["developer-setup.ipynb", *re.findall(r"^\s*- ((?:overview|tutorials)/[\w-]+\.ipynb)\s*$", guide, re.M)]:
        destination = downloads / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "guide" / name, destination)
    for source in (ROOT / "guide/examples").glob("*.ipynb"):
        destination = downloads / "examples" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in [*registry(), "data/fits/README.md"]:
        destination = ROOT / "guide/templates" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    print(f"Staged {len(names)} templates with Colab links to main.")


if __name__ == "__main__":
    stage()
