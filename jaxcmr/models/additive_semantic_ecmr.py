"""eCMR with additive semantic associations.

This variant leaves the emotional encoding machinery in
:mod:`jaxcmr.models.ecmr` unchanged and adds pre-experimental semantic
support during retrieval.
"""



from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.factory import (
    build_trial_connections_from_similarity,
    global_item_ids_by_code,
    source_categories_by_item,
)
from jaxcmr.components.response import PositionalStop
from jaxcmr.dataset import _require_pres_itemids
from jaxcmr.math import FLOAT32_EPS, relative_power_scale
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
    Parameters,
    RecallDataset,
    ResponsePolicyCreateFn,
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
        parameters: Parameters,
        is_emotional: Bool[Array, " item_count"],
        connections: Float[Array, " item_count item_count"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        response_policy_create_fn: ResponsePolicyCreateFn[eCMR] = PositionalStop,
    ) -> None:
        """Initialize additive semantic eCMR."""
        super().__init__(
            list_length,
            parameters,
            is_emotional,
            mfc_create_fn,
            mcf_create_fn,
            context_create_fn,
            response_policy_create_fn,
        )
        self.semantic_scale = parameters["semantic_scale"]
        self.msem = connections * self.semantic_scale

    def _semantic_evidence(self) -> Float[Array, " items"]:
        return lax.cond(
            self.recall_total == 0,
            lambda: jnp.zeros_like(self.msem[0]),
            lambda: self.msem[self.recalls[self.recall_total - 1] - 1],
        )

    def item_evidence(self) -> Float[Array, " items"]:
        """Add the last recall's semantic cue to absolute memory evidence."""
        return super().item_evidence() + self._semantic_evidence()

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Compute item weights for the selected studied candidates."""
        candidates = candidates & self.studied
        # Historical relative-power choice masks memory before adding semantics.
        combined = super().item_evidence() * candidates + self._semantic_evidence()
        return (relative_power_scale(combined, sensitivity) + FLOAT32_EPS) * candidates


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    response_policy_create_fn: ResponsePolicyCreateFn[eCMR] = PositionalStop,
) -> type[MemorySearchModelFactory]:
    """Add semantic support to emotional and temporal support before weighting candidates.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Construct the memory probed to reinstate context.
    mcf_create_fn : MemoryCreateFn
        Construct the memory probed for candidate support.
    context_create_fn : ContextCreateFn
        Construct the temporal context for a trial.
    response_policy_create_fn : ResponsePolicyCreateFn[eCMR]
        The policy assigns probability to STOP and each recorded item.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class initialized with the dataset and any semantic inputs.
    """

    class AdditiveSemanticECMRModelFactory:
        """Factory creating trial-specific additive semantic eCMR instances."""

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
                return AdditiveSemanticECMR(
                    list_length,
                    parameters,
                    is_emotional,
                    connections,
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

    return AdditiveSemanticECMRModelFactory
