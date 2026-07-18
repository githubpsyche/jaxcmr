"""eCMR with additive semantic associations and emotional target bias."""

from typing import Mapping, Optional, Type

import numpy as np
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.factory import build_trial_connections_from_similarity
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.models.additive_semantic_ecmr import AdditiveSemanticECMR
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
    "AdditiveSemanticRetrievalBiasECMR",
    "make_factory",
]


class AdditiveSemanticRetrievalBiasECMR(AdditiveSemanticECMR):
    """Semantic eCMR with cue-unconditional emotional target bias."""

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
        """Initialize semantic eCMR with retrieval-stage target bias."""
        super().__init__(
            list_length,
            parameters,
            is_emotional,
            connections,
            mfc_create_fn,
            mcf_create_fn,
            context_create_fn,
            termination_policy_create_fn,
        )
        self.emotional_retrieval_bias = parameters["emotional_retrieval_bias"]

    def candidate_activations(
        self, candidates: Bool[Array, " item_count"]
    ) -> Float[Array, " item_count"]:
        """Compute activations with a multiplicative emotional target bias."""
        base_activations = super().candidate_activations(candidates)
        retrieval_bias = jnp.exp(self.emotional_retrieval_bias * self.is_emotional)
        return base_activations * retrieval_bias

    def activations(self) -> Float[Array, " item_count"]:
        """Compute activations with a multiplicative emotional target bias."""
        return self.candidate_activations(self.recallable)


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
) -> Type[MemorySearchModelFactory]:
    """Build an emotional-target-bias semantic eCMR factory."""

    class AdditiveSemanticRetrievalBiasECMRModelFactory:
        """Factory creating trial-specific retrieval-bias semantic eCMR instances."""

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
                return AdditiveSemanticRetrievalBiasECMR(
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

    return AdditiveSemanticRetrievalBiasECMRModelFactory
