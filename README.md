# jaxcmr

JAXCMR is a Python library for computational models of memory search, including
Context Maintenance and Retrieval (CMR) models. It supports specifying study
events, generating recall predictions, analysing observed and simulated
behavior, and estimating model parameters using JAX.

## Installation

The instructions below target the planned **0.2.0** release. This checkout is
currently **0.2.0rc1**; installation of 0.2.0 from PyPI still needs to be confirmed
before these instructions are ready to use. For the current checkout, use the
separate development instructions below.

With Python 3.12 in your chosen environment:

```sh
python -m pip install "jaxcmr==0.2.0"
```

For a complete Apple Silicon Mac setup, follow
[Run JAXCMR in VS Code](guide/getting-started.qmd). It covers Python installation,
VS Code, notebook kernel selection and an environment check. No workshop ZIP,
source checkout or research dataset is needed for that published-package route.

## Documentation and examples

Start with the [guide introduction](guide/index.qmd) for how data, analyses,
simulation, likelihood and fitting connect. The [Templates index](guide/templates.qmd)
links fourteen complete procedures using data from the war-photo-recall
experiment, with downloadable notebooks and shared inputs. They cover behavioral
analysis, fitting, simulation, parameter exploration and pooled model comparison.

The authoritative notebooks remain in [templates/](templates/). Other notebooks
in that collection are research examples with varying dependencies and
maintenance status; they are outside the prepared book chapters.

## Development and testing

From the repository root, using its Python environment:

```sh
python -m pip install -e ".[test]"
```

See [TESTING.md](TESTING.md) for the existing test commands and instructions for
building the guide. The `test` extra includes the notebook kernel and test tools.

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
