# Testing JAXCMR

Use Python 3.12 or later. From the repository root:

```sh
python -m pip install -e ".[test]"
python -m pytest
```

For the locked development environment:

```sh
uv sync --locked --no-dev --extra test
uv run --no-sync python -m pytest
```

The complete command includes scientific regressions, seven isolated deliberate
faults, package installation, the README example, and the CMR notebook. It needs
internet access to resolve the package's declared dependencies in a new pip
environment. Each notebook run starts a fresh kernel outside the checkout.
There is no private dataset requirement and no optional regression collection.

CI runs on Linux and Apple-silicon macOS with Python 3.12. Scientific checks use
`uv.lock`; distribution checks build a wheel from the source archive and install
it with ordinary pip resolution. The two jobs split the same required collection.
Neither job is allowed to fail silently. Repository branch protection, if used,
should require both jobs on both operating systems.

While editing, select the affected file or scenario, for example:

```sh
python -m pytest tests/test_transitions.py
```

Run the complete collection before submitting a change. Runtime is diagnostic,
not a target. Do not weaken scientific checks to make a run faster.

## Adding or changing a scenario

**Every new scenario must identify an incorrect behavior it catches that existing
scenarios do not.** Put the expected answer and its short explanation beside the
input. Prefer counted probabilities, analytic losses, finite differences, and
actual user results. A shape, type, or finite-value assertion alone usually adds
little protection.

Parameterize independent examples so each is visible in collection. Multiple
assertions can describe one result; loops can follow one sequence of study events
or an A/B/A selection regression. Do not hide unrelated inputs in loops or move
assertions into an elaborate helper system. Share a small dataset or parameter
fixture when it removes real duplication.

The synthetic dataset and the retained numerical reference are documented in
[tests/fixtures/README.md](tests/fixtures/README.md). Never skip a required
regression because a private file is absent.

[testing-audit.json](testing-audit.json) is the one-time removal/replacement record
for the 743-case collection. It is historical evidence, not a manifest to update
for every future test. The fault checks temporarily replace behavior in child
processes; they never modify library files.

## Editing the guide

The independent Quarto book in `guide/` contains the introduction, setup page,
Templates index and fourteen prepared notebook chapters. Use [Quarto](https://quarto.org/docs/download/)
1.10.20 or newer; earlier versions have an [upstream bug](https://github.com/quarto-dev/quarto-cli/issues/13583)
that breaks Show All Code / Hide All Code when copy buttons are enabled. The current
preview was rendered with 1.11.9. To build the guide:

```sh
bash build-docs.sh
```

Open `_site/guide/index.html` for the preview. This command refreshes the generated
`guide/templates/` and `guide/downloads/` staging directories and `_site/guide/` output, so retired chapters
do not remain in the site. It does not execute notebook cells.

The `book.chapters` list is the publication list. Keep notebook entries as plain
`- templates/name.ipynb` lines so the build can stage exactly those files. Edit the
authoritative notebooks in `templates/`, never the generated copies in `guide/`.
Run a changed notebook in a working folder with the documented `data/` inputs,
retain useful outputs, and review its rendered chapter and download. This is
example preparation; no automated documentation suite is added.

The other templates, their outputs, scientific tests and research directories stay
in place. The old CMR notebook's existing test checks execution, not teaching quality.
The source-package rules exclude generated guide inputs.

The setup page targets the planned 0.2.0 release. Until its exact PyPI command
and notebook cell succeed in a fresh environment after publication, keep the
release-status note visible. Building the site or installing a local candidate
does not establish that the published-package instructions are ready.
