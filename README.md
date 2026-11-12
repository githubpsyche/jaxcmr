# jaxcmr

JAXCMR is a Python library for computational models of memory search, including
Context Maintenance and Retrieval (CMR) models. It supports specifying study
events, generating recall predictions, analysing observed and simulated
behavior, and estimating model parameters using JAX.

## Installation

With Python 3.12 in your chosen environment:

```sh
pip install git+https://github.com/githubpsyche/jaxcmr
```

## Documentation and examples

Start with the [guide introduction](guide/index.ipynb) for how data, analyses,
simulation, likelihood and fitting connect. The [Templates index](guide/templates.qmd)
links fourteen complete procedures using data from the war-photo-recall
experiment, with downloadable notebooks and shared inputs. They cover behavioral
analysis, fitting, simulation, parameter exploration and pooled model comparison.

The templates connect a [single configurable fitting procedure](templates/fitting.ipynb)
to saved simulations, interventions and comparisons. Analysis notebooks read the supplied
response files or your own saved inputs.

Use **Open in Colab** in a prepared chapter to install JAXCMR from GitHub’s `main` branch
and fetch its example inputs automatically; no PyPI release is required for this route.
See [the Colab instructions](guide/templates.qmd#run-in-colab).

The authoritative notebooks remain in [templates/](templates/). Other notebooks
in that collection are research examples with varying dependencies and
maintenance status; they are outside the prepared book chapters.

## Development and testing

From the repository root:

```sh
uv sync
```

The repository selects Python 3.12 and installs the default notebook and development
tools. Follow the [Local setup with VS Code](guide/developer-setup.ipynb) for VS Code and kernel
selection. See [Development](guide/development.qmd) for running tests, adding
regressions and editing the documentation. Embeddings, profiling and
Numba tooling are opt-in dependency groups or extras described in the
[development guide](guide/development.qmd#optional-tools).

## Paper draft

The working manuscript is [paper/index.qmd](paper/index.qmd). From the repository
root, run `quarto render paper` to write `_site/paper/index.html`, or
`quarto preview paper` for a live preview. Rendering does not execute code.
Supporting research records remain local and are not included with this draft.

## Quick Start

```python
from jaxcmr.models.cmr import CMR

params = {
    "encoding_drift_rate": 0.7,
    "start_drift_rate": 0.6,
    "recall_drift_rate": 0.85,
    "learning_rate": 0.4,
    "primacy_scale": 8.0,
    "primacy_decay": 1.0,
    "item_support": 6.0,
    "shared_support": 0.02,
    "choice_sensitivity": 1.5,
    "stop_probability_scale": 0.005,
    "stop_probability_growth": 0.4,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}

model = CMR(list_length=16, parameters=params)

# Study 16 items
for i in range(16):
    model = model.experience_item(i)

# Retrieve
model = model.start_retrieving()
probs = model.outcome_probabilities()  # stop prob + 16 item probs
```
