"""eCMR variant with source-switch temporal disruption.

This module leaves the base :mod:`jaxcmr.models.ecmr` behavior unchanged and
adds a separate factory path for Talmi-style source switches. On transitions
between neutral and emotional study items, the model can drift temporal context
toward a fresh out-of-list unit before encoding the next item.
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
from jaxcmr.models.ecmr import eCMR
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
    "eCMRSourceSwitch",
    "make_factory",
]

_init_expanded_context = TemporalContext.TemporalContext.init_expanded


def _zero_temporal_outlist_mcf_rows(
    memory: LinearMemory.LinearMemory,
    item_count: int,
) -> LinearMemory.LinearMemory:
    """Clear pre-existing item support from expanded temporal outlist rows."""
    return memory.replace(state=memory.state.at[item_count + 1 :].set(0.0))


class eCMRSourceSwitch(eCMR):
    """eCMR with optional emotional/neutral source-switch disruption."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        is_emotional: Bool[Array, " item_count"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: OutlistContextCreateFn = _init_expanded_context,
        response_policy_create_fn: ResponsePolicyCreateFn[eCMR] = PositionalStop,
    ) -> None:
        """Initialize eCMR with expanded temporal context for switch events."""
        super().__init__(
            list_length,
            parameters,
            is_emotional,
            mfc_create_fn,
            mcf_create_fn,
            context_create_fn,
            response_policy_create_fn,
        )
        self.source_switch_drift_rate = parameters.get("source_switch_drift_rate", 0.0)
        self.source_categories = self.is_emotional.astype(jnp.int32)
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
        return super(eCMRSourceSwitch, switched_model).experience_item(item_index)


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: OutlistContextCreateFn = _init_expanded_context,
    response_policy_create_fn: ResponsePolicyCreateFn[eCMR] = PositionalStop,
) -> type[MemorySearchModelFactory]:
    """Drift toward fresh context when the emotional category changes during study.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Construct the memory probed to reinstate context.
    mcf_create_fn : MemoryCreateFn
        Construct the memory probed for candidate support.
    context_create_fn : OutlistContextCreateFn
        Use a context with room for out-of-list units.
    response_policy_create_fn : ResponsePolicyCreateFn[eCMR]
        The policy assigns probability to STOP and each recorded item.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class initialized with the dataset and any semantic inputs.
    """

    class eCMRSourceSwitchModelFactory:
        """Factory creating trial-specific eCMR source-switch instances."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.present_lists = jnp.array(_require_pres_itemids(dataset))
            self.trial_emotions = source_categories_by_item(dataset, signed=False) != 0
            self.max_list_length = self.trial_emotions.shape[1]

            def model_create_fn(
                list_length: int,
                parameters: Parameters,
                is_emotional: Bool[Array, " item_count"],
            ) -> MemorySearch:
                return eCMRSourceSwitch(
                    list_length,
                    parameters,
                    is_emotional,
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
                self.trial_emotions[0],
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
                self.trial_emotions[trial_index],
            )

    return eCMRSourceSwitchModelFactory
