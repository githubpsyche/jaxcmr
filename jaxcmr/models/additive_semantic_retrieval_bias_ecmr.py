"""eCMR with additive semantic associations and emotional target bias."""



from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.factory import (
    build_trial_connections_from_similarity,
    global_item_ids_by_code,
    source_categories_by_item,
)
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.dataset import _require_pres_itemids
from jaxcmr.models.additive_semantic_ecmr import AdditiveSemanticECMR
from jaxcmr.models.ecmr import eCMR
from jaxcmr.typing import (
    Array,
    Bool,
    ContextCreateFn,
    Float,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    Parameters,
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
        parameters: Parameters,
        is_emotional: Bool[Array, " item_count"],
        connections: Float[Array, " item_count item_count"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn[
            eCMR
        ] = PositionalTermination,
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
    termination_policy_create_fn: TerminationPolicyCreateFn[
        eCMR
    ] = PositionalTermination,
) -> type[MemorySearchModelFactory]:
    """Build an emotional-target-bias semantic eCMR factory."""

    class AdditiveSemanticRetrievalBiasECMRModelFactory:
        """Factory creating trial-specific retrieval-bias semantic eCMR instances."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.present_lists = jnp.array(_require_pres_itemids(dataset))
            self.trial_emotions = source_categories_by_item(dataset, signed=False) != 0
            self.max_list_length = self.trial_emotions.shape[1]
            vocabulary_ids = global_item_ids_by_code(
                dataset["pres_itemnos"], self.present_lists, self.max_list_length
            )
            self.trial_connections = build_trial_connections_from_similarity(vocabulary_ids, features)

            def model_create_fn(
                list_length: int,
                parameters: Parameters,
                is_emotional: Bool[Array, " item_count"],
                connections: Float[Array, " item_count item_count"],
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

        def create_model(self, parameters: Parameters) -> MemorySearch:
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
            parameters: Parameters,
        ) -> MemorySearch:
            """Create model for a specific trial."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_emotions[trial_index],
                self.trial_connections[trial_index],
            )

    return AdditiveSemanticRetrievalBiasECMRModelFactory
