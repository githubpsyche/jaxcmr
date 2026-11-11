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

The rendered book is committed in `docs/`. GitHub Pages publishes that folder
from `main` at https://githubpsyche.github.io/jaxcmr/. After changing the guide,
prepare and review a local preview before committing and publishing the source and `docs/`.
Notebook cells are not executed during rendering.

The independent Quarto book in `guide/` contains 26 content pages: the introduction,
setup page, Library Guide introduction and eight topic notebooks, Templates index and
fourteen prepared template notebooks. Use [Quarto](https://quarto.org/docs/download/)
1.10.20 or newer; earlier versions have an [upstream bug](https://github.com/quarto-dev/quarto-cli/issues/13583)
that breaks Show All Code / Hide All Code when copy buttons are enabled. The current
preview was rendered with 1.11.9. To build the guide:

```sh
bash build-docs.sh
```

Open `docs/index.html` for the preview. This command refreshes the generated
`guide/templates/` and `guide/downloads/` staging directories and `docs/` output, so retired chapters
do not remain in the site. It does not execute notebook cells.

### Prepare matching Colab revisions

The default build permits uncommitted local work and omits Colab links when a matching
committed notebook/package pair is unavailable. It never commits or pushes changes.

After review, commit the package, dependency and registered input changes first. Then:

```sh
uv run python prepare-guide.py pin
```

This reads the current commit and updates the fourteen conditional Colab setup pins.
Review and commit those notebooks. Prepare the publishable book with:

```sh
bash build-docs.sh --publish
```

The build derives immutable Colab URLs from the last notebook commit, stages its exact
notebooks as downloads, and rejects inconsistent pins, uncommitted notebook/package/input
changes, or incorrect input hashes. Generated presentation links are added only to the
staged chapters. Commit `docs/` after review and publish the source commits with it;
there is no manually maintained Colab tag. A local preview's unpublished Colab setup
cell reports that preparation is required before installing anything.

The `book.chapters` list is the publication list. Keep notebook entries as plain
`- templates/name.ipynb` or `- overview/name.ipynb` lines. Authoritative template
notebooks live in `templates/`; their copies in `guide/templates/` are generated.
Authoritative Library Guide pages live in `guide/overview/`. The build stages notebook
downloads for both sections under `guide/downloads/`; do not edit those copies.
Run a changed notebook in a working folder with the documented `data/` inputs,
retain useful outputs, and review its rendered chapter and download. This is
example preparation; no automated documentation suite is added.

The other templates, their outputs, scientific tests and research directories stay
in place. The old CMR notebook's existing test checks execution, not teaching quality.
The source-package rules exclude generated guide inputs.

## Model-comparison API migration

See [MODEL_COMPARISON_MIGRATION.md](MODEL_COMPARISON_MIGRATION.md) for the NLL
convention, paired identifier requirements, changed interval returns, separate
nested-model reporting adjustments, and the external caller inventory. Historical
research outputs are not regenerated by a library correction. Focused numerical
checks live in `tests/test_summary.py`; notebook preparation remains manual.
