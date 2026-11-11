"""Prepare notebook pins and stage the Quarto book from matching source revisions."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PIN = re.compile(r'^([ \t]*)revision = (?:"[0-9a-f]{40}"|None)(?:[ \t]+#.*)?$', re.M)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def chapters():
    return re.findall(
        r"^\s*- (templates/\w+\.ipynb)\s*$",
        (ROOT / "guide/_quarto.yml").read_text(), re.M,
    )


def registry():
    module = ast.parse((ROOT / "jaxcmr/colab.py").read_text())
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_INPUT_HASHES"
                              for target in node.targets))
    return ast.literal_eval(assignment.value)


def package_paths():
    return ["jaxcmr", "pyproject.toml", "uv.lock", ".python-version", *registry()]


def require_matches(revision, paths):
    changed = git("diff", "--name-only", revision, "--", *paths).decode().splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths).decode().splitlines()
    if changed or untracked:
        raise ValueError("Commit the reviewed changes first: " + ", ".join(changed + untracked))


def require_inputs():
    for name, expected in registry().items():
        with (ROOT / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Example input differs from its registry hash: {name}")


def package_pin(notebook):
    setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                 and '"google.colab" in sys.modules' in "".join(cell["source"]))
    matches = PIN.findall("".join(setup["source"]))
    if len(matches) != 1:
        raise ValueError("Expected one package revision in the conditional Colab cell.")
    match = re.search(r'revision = "([0-9a-f]{40})"', "".join(setup["source"]))
    return match.group(1) if match else None


def pin_notebooks():
    revision = git("rev-parse", "HEAD").decode().strip()
    require_matches(revision, package_paths())
    require_inputs()
    for name in chapters():
        path = ROOT / name
        original = path.read_bytes()
        notebook = json.loads(original)
        package_pin(notebook)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and '"google.colab" in sys.modules' in "".join(cell["source"]):
                source = PIN.sub(lambda match: f'{match[1]}revision = "{revision}"', "".join(cell["source"]))
                cell["source"] = source.splitlines(keepends=True)
        if path.read_bytes() != original:
            raise ValueError(f"Concurrent notebook edit: {name}")
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Pinned {len(chapters())} notebooks to {revision}. Review and commit the notebooks next.")


def publication_revision(names):
    revision = git("log", "-1", "--format=%H", "--", *names).decode().strip()
    require_matches(revision, names)
    pins = {package_pin(json.loads(git("show", f"{revision}:{name}"))) for name in names}
    if len(pins) != 1 or None in pins:
        raise ValueError("Run prepare-guide.py pin and commit all prepared notebooks.")
    package_revision = pins.pop()
    require_matches(package_revision, package_paths())
    if git("diff", "--name-only", package_revision, revision, "--", *package_paths()).strip():
        raise ValueError("Notebook revision contains package/input changes after its pinned package revision.")
    require_inputs()
    return revision


def stage(publish):
    names = chapters()
    if not names:
        raise ValueError("No template chapters found in guide/_quarto.yml.")
    try:
        revision = publication_revision(names)
    except (ValueError, subprocess.CalledProcessError) as error:
        if publish:
            raise ValueError(f"Publication stopped: {error}") from error
        revision = None
        print(f"Local preview: Colab links omitted. {error}")

    for folder in ["guide/templates", "guide/downloads", "docs"]:
        path = ROOT / folder
        if path.is_symlink():
            raise ValueError(f"Generated output must not be a symlink: {folder}")
        if path.exists():
            shutil.rmtree(path)
    downloads = ROOT / "guide/downloads"
    downloads.mkdir(parents=True)
    for name in names:
        raw = git("show", f"{revision}:{name}") if revision else (ROOT / name).read_bytes()
        (downloads / Path(name).name).write_bytes(raw)
        notebook = json.loads(raw)
        if revision:
            front = notebook["cells"][0]
            source = "".join(front["source"])
            link = ("      - text: Open in Colab\n"
                    "        icon: box-arrow-up-right\n"
                    f"        href: https://colab.research.google.com/github/githubpsyche/jaxcmr/blob/{revision}/{name}\n"
                    "        target: _blank\n        rel: noopener\n")
            source = source.rstrip().removesuffix("---") + link + "---\n"
            front["source"] = source.splitlines(keepends=True)
        destination = ROOT / "guide" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")

    guide = (ROOT / "guide/_quarto.yml").read_text()
    for name in ["developer-setup.ipynb", *re.findall(r"^\s*- (overview/\w+\.ipynb)\s*$", guide, re.M)]:
        destination = downloads / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "guide" / name, destination)
    for name in [*registry(), "data/fits/README.md"]:
        destination = ROOT / "guide/templates" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    print(f"Staged {len(names)} templates" + (f" from {revision}." if revision else " for local review."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pin", help="Pin all Colab setup cells to the current committed package.")
    staging = commands.add_parser("stage", help="Stage the book and notebook downloads.")
    staging.add_argument("--publish", action="store_true", help="Require matching committed notebooks, package and inputs.")
    arguments = parser.parse_args()
    try:
        pin_notebooks() if arguments.command == "pin" else stage(arguments.publish)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
