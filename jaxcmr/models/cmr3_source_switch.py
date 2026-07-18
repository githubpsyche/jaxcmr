"""CMR3 variant with source-switch temporal disruption.

This module leaves the base :mod:`jaxcmr.models.cmr3` behavior unchanged and
adds a separate factory path for valence-category source switches. On
transitions among negative, neutral, and positive study items, the model can
drift temporal context toward a fresh out-of-list unit before encoding the next
item.
"""

from typing import Mapping, Optional, Type

import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.models.cmr3 import CMR3
from jaxcmr.typing import (
    Array,
    Bool,
    ContextCreateFn,
    Float,
    Float_,
    Int_,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
    TerminationPolicyCreateFn,
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
        parameters: Mapping[str, Float_],
        is_negative: Bool[Array, " study_events"],
        is_positive: Bool[Array, " study_events"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = _init_expanded_context,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
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
            termination_policy_create_fn,
        )
        self.source_switch_drift_rate = parameters.get("source_switch_drift_rate", 0.0)
        self.source_categories = (
            self.is_positive.astype(jnp.int32) - self.is_negative.astype(jnp.int32)
        )
        self.mcf = _zero_temporal_outlist_mcf_rows(self.mcf, self.item_count)

    def _should_apply_source_switch(self) -> Bool[Array, ""]:
        """Return whether the next study item changes valence source category."""
        current_category = self.source_categories[self.study_index]
        previous_index = jnp.maximum(self.study_index - 1, 0)
        previous_category = self.source_categories[previous_index]
        category_changed = current_category != previous_category
        return jnp.logical_and(
            self.study_index > 0,
            jnp.logical_and(category_changed, self.source_switch_drift_rate > 0),
        )

    def _apply_source_switch_context(self) -> TemporalContext.TemporalContext:
        """Drift temporal context toward a fresh unlearned outlist unit."""
        return self.context.integrate_with_outlist(
            self.context.zeros,
            ratio=0.0,
            drift_rate=self.source_switch_drift_rate,
        )

    def experience_item(self, item_index: Int_) -> "CMR3SourceSwitch":
        """Optionally apply source-switch disruption before item encoding."""
        switched_context = lax.cond(
            self._should_apply_source_switch(),
            self._apply_source_switch_context,
            lambda: self.context,
        )
        return CMR3.experience_item(self.replace(context=switched_context), item_index)


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = _init_expanded_context,
    termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
) -> Type[MemorySearchModelFactory]:
    """Build a CMR3 source-switch factory."""

    class CMR3SourceSwitchModelFactory:
        """Factory creating trial-specific CMR3 source-switch instances."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ) -> None:
            self.present_lists = jnp.array(dataset["pres_itemids"])
            self.max_list_length = np.max(dataset["listLength"]).item()

            valence = jnp.array(dataset["valence"])
            self.trial_negative = jnp.array(valence < 0, dtype=bool)
            self.trial_positive = jnp.array(valence > 0, dtype=bool)

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_negative: Bool[Array, " study_events"],
                is_positive: Bool[Array, " study_events"],
            ) -> MemorySearch:
                return CMR3SourceSwitch(
                    list_length,
                    parameters,
                    is_negative,
                    is_positive,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    termination_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Mapping[str, Float_]) -> MemorySearch:
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
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            """Create model for a specific trial."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_negative[trial_index],
                self.trial_positive[trial_index],
            )

    return CMR3SourceSwitchModelFactory
