"""CMR3 variant with source-switch temporal disruption.

This module leaves the base :mod:`jaxcmr.models.cmr3` behavior unchanged and
adds a separate factory path for valence-category source switches. On
transitions among negative, neutral, and positive study items, the model can
drift temporal context toward a fresh out-of-list unit before encoding the next
item.
"""



from typing import Self, cast

from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.context import OutlistContext, OutlistContextCreateFn
from jaxcmr.components.factory import source_categories_by_item
from jaxcmr.components.response import PositionalStop
from jaxcmr.dataset import _require_pres_itemids
from jaxcmr.models.cmr3 import CMR3
from jaxcmr.typing import (
    Array,
    Bool,
    Context,
    Float,
    Int_,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    Parameters,
    RecallDataset,
    ResponsePolicyCreateFn,
)

__all__ = [
    "CMR3SourceSwitch",
    "make_factory",
]

_init_expanded_context = TemporalContext.TemporalContext.init_expanded


def _zero_temporal_outlist_mcf_rows(
    memory: LinearMemory.LinearMemory,
    item_count: int,
) -> LinearMemory.LinearMemory:
    """Clear pre-existing item support from expanded temporal outlist rows."""
    return memory.replace(state=memory.state.at[item_count + 1 :].set(0.0))


class CMR3SourceSwitch(CMR3):
    """CMR3 with optional negative/neutral/positive source-switch disruption."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        is_negative: Bool[Array, " item_count"],
        is_positive: Bool[Array, " item_count"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: OutlistContextCreateFn = _init_expanded_context,
        response_policy_create_fn: ResponsePolicyCreateFn[CMR3] = PositionalStop,
    ) -> None:
        """Initialize CMR3 with expanded temporal context for switch events."""
        super().__init__(
            list_length,
            parameters,
            is_negative,
            is_positive,
            mfc_create_fn,
            mcf_create_fn,
            context_create_fn,
            response_policy_create_fn,
        )
        self.source_switch_drift_rate = parameters.get("source_switch_drift_rate", 0.0)
        self.source_categories = (
            self.is_positive.astype(jnp.int32) - self.is_negative.astype(jnp.int32)
        )
        self.previous_source_category = jnp.int32(0)
        self.mcf = _zero_temporal_outlist_mcf_rows(self.mcf, self.item_count)

    def _apply_source_switch_context(self) -> Context:
        """Drift temporal context toward a fresh unlearned outlist unit."""
        # The base constructor stores this specialized context as Context.
        return cast(OutlistContext, self.context).integrate_with_outlist(
            jnp.zeros_like(self.context.state),
            ratio=0.0,
            drift_rate=self.source_switch_drift_rate,
        )

    def experience_item(self, item_index: Int_) -> Self:
        """Optionally apply source-switch disruption before item encoding."""
        category = self.source_categories[item_index]
        switch = (
            (self.study_index > 0)
            & (category != self.previous_source_category)
            & (self.source_switch_drift_rate > 0)
        )
        switched_context = lax.cond(
            switch,
            self._apply_source_switch_context,
            lambda: self.context,
        )
        switched_model = self.replace(
            context=switched_context, previous_source_category=category
        )
        return super(CMR3SourceSwitch, switched_model).experience_item(item_index)


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: OutlistContextCreateFn = _init_expanded_context,
    response_policy_create_fn: ResponsePolicyCreateFn[CMR3] = PositionalStop,
) -> type[MemorySearchModelFactory]:
    """Drift toward fresh context when successive study items change valence category.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn, optional
        Construct temporal item-to-context memory.
    mcf_create_fn : MemoryCreateFn, optional
        Construct temporal context-to-item memory.
    context_create_fn : OutlistContextCreateFn, optional
        Construct temporal context with units reserved for source switches.
    response_policy_create_fn : ResponsePolicyCreateFn[CMR3], optional
        Assign probabilities to STOP and the items using the current source
        and temporal context. The default schedules STOP by recall count.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class that reads each studied item's valence from the dataset.
    """

    class CMR3SourceSwitchModelFactory:
        """Factory creating trial-specific CMR3 source-switch instances."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.present_lists = jnp.array(_require_pres_itemids(dataset))

            valence = source_categories_by_item(dataset, signed=True)
            self.max_list_length = valence.shape[1]
            self.trial_negative = valence < 0
            self.trial_positive = valence > 0

            def model_create_fn(
                list_length: int,
                parameters: Parameters,
                is_negative: Bool[Array, " item_count"],
                is_positive: Bool[Array, " item_count"],
            ) -> MemorySearch:
                return CMR3SourceSwitch(
                    list_length,
                    parameters,
                    is_negative,
                    is_positive,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    response_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Parameters) -> MemorySearch:
            """Create model from first trial for shape inference."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_negative[0],
                self.trial_positive[0],
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            """Create model for a specific trial."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_negative[trial_index],
                self.trial_positive[trial_index],
            )

    return CMR3SourceSwitchModelFactory
