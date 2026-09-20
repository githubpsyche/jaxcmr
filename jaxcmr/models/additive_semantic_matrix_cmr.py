"""CMR with additive semantic associations from a similarity matrix."""

from collections.abc import Mapping

import numpy as np

from jaxcmr.components.factory import build_trial_connections_from_similarity
from jaxcmr.helpers import require_pres_itemids
from jaxcmr.models.additive_semantic_cmr import CMR
from jaxcmr.typing import (
    Array,
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
    "CMR",
    "make_factory",
]


def make_factory(
    mfc_create_fn: MemoryCreateFn,
    mcf_create_fn: MemoryCreateFn,
    context_create_fn: ContextCreateFn,
    termination_policy_create_fn: TerminationPolicyCreateFn[CMR],
) -> type[MemorySearchModelFactory]:
    class CMRModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items word_pool_items"] | None,
        ):
            self.present_lists = np.array(require_pres_itemids(dataset))
            self.max_list_length = np.max(dataset["listLength"]).item()
            self.trial_connections = build_trial_connections_from_similarity(
                self.present_lists,
                features,
            )

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                connections: Float[Array, "study_events study_events"],
            ) -> MemorySearch:
                return CMR(
                    list_length,
                    parameters,
                    connections,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    termination_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Mapping[str, Float_]) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_connections[0],
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_connections[trial_index],
            )

    return CMRModelFactory
