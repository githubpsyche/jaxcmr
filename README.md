# jaxcmr

JAXCMR is a JAX-accelerated toolkit for modeling memory search, featuring implementations of Context Maintenance and Retrieval (CMR) models and variants for free/serial recall. It includes utilities for simulation, likelihood evaluation, fitting, and common behavioral analyses.

## Installation

```bash
pip install jaxcmr
```

Python 3.12+ is required. The `jax` dependency installs automatically; consult JAX documentation if you need a specific accelerator build.

## Quick start

Simulate a single free-recall trial with a basic CMR model:

```python
from jax import random
import jax.numpy as jnp
from jaxcmr.models.cmr import BaseCMR
from jaxcmr.simulation import simulate_study_and_free_recall

# Toy CMR parameters (example values)
params = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.5,
    "recall_drift_rate": 0.5,
    "shared_support": 0.05,
    "item_support": 0.25,
    "primacy_scale": 1.0,
    "primacy_decay": 0.6,
    "learning_rate": 0.5,
    "stop_probability_scale": 0.05,
    "stop_probability_growth": 0.2,
    "choice_sensitivity": 0.6,
}

list_length = 5
present = jnp.arange(1, list_length + 1)  # [1,2,3,4,5]
rng = random.PRNGKey(0)

model = BaseCMR(list_length, params)
model, recalls = simulate_study_and_free_recall(model, present, rng)
print(recalls)  # e.g., array of recalled within-list item numbers
```

Inspect outcome probabilities step-by-step:

```python
from jax import lax

# Study the list, then enter retrieval mode
model = BaseCMR(list_length, params)
model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
model = model.start_retrieving()

# Probabilities for stopping (index 0) and recalling each item (1..list_length)
p0_and_items = model.outcome_probabilities()
print(p0_and_items)

# After retrieving an item (e.g., item 5), update the model and recompute
model = model.retrieve(5)
print(model.outcome_probabilities())
```

## Documentation and Issues

- Docs and examples: https://githubpsyche.github.io/jaxcmr/
- Source and issues: https://github.com/githubpsyche/jaxcmr
