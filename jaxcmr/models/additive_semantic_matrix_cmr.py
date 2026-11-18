"""CMR with additive semantic associations from a similarity matrix."""



import jax.numpy as jnp
import numpy as np

from jaxcmr.components.factory import (
    build_trial_connections_from_similarity,
    global_item_ids_by_code,
)
from jaxcmr.dataset import _require_pres_itemids
from jaxcmr.models.additive_semantic_cmr import CMR
from jaxcmr.typing import (
    Array,
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
        ) -> None:
            self.present_lists = jnp.asarray(_require_pres_itemids(dataset))
            self.max_list_length = np.max(dataset["listLength"]).item()
            vocabulary_ids = global_item_ids_by_code(
                dataset["pres_itemnos"], self.present_lists, self.max_list_length
            )
            self.trial_connections = build_trial_connections_from_similarity(
                vocabulary_ids,
                features,
            )

            def model_create_fn(
                list_length: int,
                parameters: Parameters,
                connections: Float[Array, "item_count item_count"],
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

        def create_model(self, parameters: Parameters) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_connections[0],
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_connections[trial_index],
            )

    return CMRModelFactory
