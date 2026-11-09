"""The small, explicit dataset shared by preparation and workflow checks."""

import json
from pathlib import Path
import jax.numpy as jnp


def recall_fixture():
    path = Path(__file__).parent / "fixtures" / "recall.json"
    return {
        key: jnp.asarray(value, dtype=jnp.int32)
        for key, value in json.loads(path.read_text()).items()
    }


CMR_PARAMS = {
    "encoding_drift_rate": 0.5,
    "start_drift_rate": 0.4,
    "recall_drift_rate": 0.6,
    "shared_support": 1.0,
    "item_support": 1.5,
    "learning_rate": 0.3,
    "primacy_scale": 1.0,
    "primacy_decay": 0.1,
    "choice_sensitivity": 2.0,
    "learn_after_context_update": False,
    "allow_repeated_recalls": False,
}
