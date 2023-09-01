"""
CMR
Type and functions for the Context Maintenance and Retrieval model.
"""

# %% Imports

from simple_pytree import Pytree, static_field
from jaxcmr.context import TemporalContext, Context
from jaxcmr.memory import (
    OneWayMemory,
    LinearAssociativeMcf,
    LinearAssociativeMfc,
    InstanceMcf,
)
from functools import partial
from plum import dispatch
from jax import jit, numpy as jnp
from beartype.typing import Callable

from jaxcmr.helpers import (
    Integer,
    Float,
    Array,
    ScalarInteger,
    ScalarFloat,
    item_features,
    item_count,
)

# %% Exports

__all__ = [
    "MemorySearch",
    "CMR",
    "BaseCMR",
    "InstanceCMR",
    "exponential_primacy_weighting",
]

# %% Base Type Hierarchy


class MemorySearch(Pytree, mutable=True):
    item_count: int  # the number of items initialized with the model
    is_active: bool  # whether model still open to new experiences or retrievals


class CMR(MemorySearch, mutable=True):
    item_count = static_field()

    def __init__(
        self,
        items: Float[Array, "item_count item_features"],
        presentation_count: ScalarInteger,
        context: Context,
        mfc: OneWayMemory,
        mcf: OneWayMemory,
        parameters: dict,
    ):
        "The Context Maintenance and Retrieval (CMR) model of memory search."

        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.delay_drift_rate = parameters["delay_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.stop_probability_scale = parameters["stop_probability_scale"]
        self.stop_probability_growth = parameters["stop_probability_growth"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self._mcf_learning_rate = exponential_primacy_weighting(
            presentation_count, parameters["primacy_scale"], parameters["primacy_decay"]
        )

        self.items = items
        self.mfc = mfc
        self.mcf = mcf
        self.context = context

        self.item_count = items.shape[0]
        self.recall_sequence = jnp.zeros(self.item_count, jnp.int32)
        self.recall_mask = jnp.zeros(self.item_count, jnp.bool_)
        self.encoding_index = 0
        self.is_active = True
        self.recall_total = 0

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        return self._mcf_learning_rate[self.encoding_index]


# %% Model Variant Constructors


@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def basic_init_cmr(
    mfc_init: Callable,
    mcf_init: Callable,
    context_init: Callable,
    item_count: ScalarInteger,
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    "Initialize CMR with linear associative memories and a temporal context."

    mfc = mfc_init(item_count, presentation_count, parameters)
    mcf = mcf_init(item_count, presentation_count, parameters)
    context = context_init(item_count)
    items = jnp.eye(item_count, item_count)
    return CMR(items, presentation_count, context, mfc, mcf, parameters)


@partial(jit, static_argnums=(0, 1, 2, 3, 5))
def generalized_init_cmr(
    mfc_init: Callable,
    mcf_init: Callable,
    context_init: Callable,
    items: Integer[Array, "item_count item_features"],
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    "Initialize CMR with linear associative memories and a temporal context."

    mfc = mfc_init(items, presentation_count, parameters)
    mcf = mcf_init(items, presentation_count, parameters)
    context = context_init(items.shape[0])
    return CMR(items, presentation_count, context, mfc, mcf, parameters)


# %% Variants


@dispatch
def BaseCMR(
    item_count: ScalarInteger,
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    return basic_init_cmr(
        LinearAssociativeMfc.create,
        LinearAssociativeMcf.create,
        TemporalContext.create,
        item_count,
        presentation_count,
        parameters,
    )


@dispatch
def BaseCMR(
    items: Integer[Array, "item_count item_features"],
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    return generalized_init_cmr(
        LinearAssociativeMfc.create,
        LinearAssociativeMcf.create,
        TemporalContext.create,
        items,
        presentation_count,
        parameters,
    )


@dispatch
def InstanceCMR(
    item_count: ScalarInteger,
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    return basic_init_cmr(
        LinearAssociativeMfc.create,
        InstanceMcf.create,
        TemporalContext.create,
        item_count,
        presentation_count,
        parameters,
    )


@dispatch
def InstanceCMR(
    items: Integer[Array, "item_count item_features"],
    presentation_count: ScalarInteger,
    parameters: dict,
) -> CMR:
    return generalized_init_cmr(
        LinearAssociativeMfc.create,
        InstanceMcf.create,
        TemporalContext.create,
        items,
        presentation_count,
        parameters,
    )


# %% Helper Functions


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
