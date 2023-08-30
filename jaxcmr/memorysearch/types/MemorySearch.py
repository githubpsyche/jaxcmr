"""
MemorySearch
Abstract type and functions for memory search models.
"""

# %% Imports

from simple_pytree import Pytree
from jaxcmr.context import Context
from jaxcmr.memory import OneWayMemory
from functools import partial
from plum import dispatch
from jax import jit, numpy as jnp

from jaxcmr.helpers import (
    replace,
    Integer,
    Bool,
    Float,
    Array,
    ScalarBool,
    ScalarInteger,
    ScalarFloat,
    item_features,
    item_count,
)

# %% Exports

__all__ = [
    "MemorySearch",
    "CMR",
]

# %% Base Type Hierarchy

class MemorySearch(Pytree, mutable=True):
    item_count: int  # the number of items initialized with the model
    is_active: bool  # whether model still open to new experiences or retrievals


class CMR(MemorySearch, mutable=True):
    mfc: OneWayMemory
    mcf: OneWayMemory
    context: Context
    encoding_drift_rate: ScalarFloat
    delay_drift_rate: ScalarFloat
    start_drift_rate: ScalarFloat
    recall_drift_rate: ScalarFloat
    mfc_learning_rate: ScalarFloat
    mcf_learning_rate: ScalarFloat
    stop_probability_scale: ScalarFloat
    stop_probability_growth: ScalarFloat
    is_active: ScalarBool
    item_count: ScalarInteger
    items: Integer[Array, "item_count item_features"]
    encoding_index: ScalarInteger
    recall_total: ScalarInteger
    recall_sequence: Integer[Array, "item_count"]
    recall_mask: Bool[Array, "item_count"]

# %% Helper functions

@partial(jit, static_argnums=(0,))
@dispatch
def exponential_primacy_weighting(
    presentation_count: ScalarInteger,
    primacy_scale: ScalarFloat,
    primacy_decay: ScalarFloat,
) -> Float[Array, "study_events"]:
    """The primacy effect as exponential decay of boosted attention weights."""
    arange = jnp.arange(presentation_count, dtype=jnp.float32)
    return primacy_scale * jnp.exp(-primacy_decay * arange) + 1