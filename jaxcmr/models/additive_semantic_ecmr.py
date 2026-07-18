"""eCMR with additive semantic associations.

This variant leaves the emotional encoding machinery in
:mod:`jaxcmr.models.ecmr` unchanged and adds pre-experimental semantic
support during retrieval.
"""

from typing import Mapping, Optional, Type

import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.factory import build_trial_connections_from_similarity
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.math import lb, power_scale
from jaxcmr.models.ecmr import eCMR
from jaxcmr.typing import (
    Array,
    Bool,
    ContextCreateFn,
    Float,
    Float_,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
    TerminationPolicyCreateFn,
)

__all__ = [
    "AdditiveSemanticECMR",
    "make_factory",
]


class AdditiveSemanticECMR(eCMR):
    """Full eCMR with additive semantic support during retrieval."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_emotional: Bool[Array, " study_events"],
        connections: Float[Array, " study_events study_events"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ) -> None:
        """Initialize additive semantic eCMR."""
        super().__init__(
            list_length,
            parameters,
            is_emotional,
            mfc_create_fn,
            mcf_create_fn,
            context_create_fn,
            termination_policy_create_fn,
        )
        self.semantic_scale = parameters["semantic_scale"]
        self.msem = connections * self.semantic_scale

    def candidate_activations(
        self, candidates: Bool[Array, " item_count"]
    ) -> Float[Array, " item_count"]:
        """Compute retrieval activations with temporal, emotional, and semantic support."""
        temporal_act = self.mcf.probe(self.context.state) * candidates
        emotional_act = self.emotion_mcf.probe(self.emotion_context.state) * candidates
        semantic_support = lax.cond(
            self.recall_total == 0,
            lambda: jnp.zeros_like(candidates, dtype=jnp.float32),
            lambda: self.msem[self.recalls[self.recall_total - 1] - 1],
        )
        combined = temporal_act + emotional_act + semantic_support
        return (power_scale(combined, self.mcf_sensitivity) + lb) * candidates

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations with temporal, emotional, and semantic support."""
        return self.candidate_activations(self.recallable)


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
) -> Type[MemorySearchModelFactory]:
    """Build an additive semantic eCMR factory."""

    class AdditiveSemanticECMRModelFactory:
        """Factory creating trial-specific additive semantic eCMR instances."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ) -> None:
            self.present_lists = jnp.array(dataset["pres_itemids"])
            self.max_list_length = np.max(dataset["listLength"]).item()
            self.trial_emotions = jnp.array(dataset["valence"] != 0, dtype=bool)
            self.trial_connections = build_trial_connections_from_similarity(
                self.present_lists, features
            )

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_emotional: Bool[Array, " study_events"],
                connections: Float[Array, " study_events study_events"],
            ) -> MemorySearch:
                return AdditiveSemanticECMR(
                    list_length,
                    parameters,
                    is_emotional,
                    connections,
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
                self.trial_emotions[0],
                self.trial_connections[0],
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
                self.trial_emotions[trial_index],
                self.trial_connections[trial_index],
            )

    return AdditiveSemanticECMRModelFactory
