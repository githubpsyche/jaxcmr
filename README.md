# jaxcmr

JAX-accelerated toolkit for modeling memory search, featuring implementations of Context Maintenance and Retrieval (CMR) models and variants for free and serial recall.

## Features

- **CMR model family** — base CMR plus repetition variants (positional, drift, reinforcement, blend, distinct, outlist, no-reinstate) and semantic variants (additive, multiplicative)
- **Behavioral analyses** — 30+ measures including SPC, CRP, lag-rank, co-recall, distance-CRP, error rates, and category-based analyses
- **Simulation** — trial-level and dataset-level simulation with `vmap`-accelerated parameter sweeps
- **Fitting** — likelihood-based fitting via SciPy differential evolution, with permutation and MSE loss alternatives
- **JAX throughout** — automatic differentiation, JIT compilation, and GPU/TPU support

## Installation

```bash
pip install jaxcmr
```

For development:

```bash
git clone https://github.com/githubpsyche/jaxcmr.git
cd jaxcmr
pip install -e ".[dev]"
```

## Documentation

Full documentation at [githubpsyche.github.io/jaxcmr](https://githubpsyche.github.io/jaxcmr), including:

- [Guide](https://githubpsyche.github.io/jaxcmr/templates/guide/) — core abstractions, JAX patterns, data format, evaluation pipeline
- [Analyses](https://githubpsyche.github.io/jaxcmr/templates/) — template notebooks for every behavioral measure
- [Models](https://githubpsyche.github.io/jaxcmr/templates/models/) — CMR walkthrough and model variants
- [Evaluation](https://githubpsyche.github.io/jaxcmr/templates/evaluation/) — loss functions, fitting, parameter sensitivity

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
}

model = CMR(list_length=16, parameters=params)

# Study 16 items
for i in range(16):
    model = model.experience_item(i)

# Retrieve
model = model.start_retrieving()
probs = model.outcome_probabilities()  # stop prob + 16 item probs
```