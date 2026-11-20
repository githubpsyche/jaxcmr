"""Multiplicative semantic CMR combines temporal support with a semantic
cue from the last recalled item. On the first recall the semantic factor
is one for each available item. On later recalls the model scales the
temporal and semantic support separately and multiplies them.
"""



import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.factory import (
    build_trial_connections,
    global_item_ids_by_code,
)
from jaxcmr.components.response import PositionalStop
from jaxcmr.dataset import _require_pres_itemids
from jaxcmr.math import (
    FLOAT32_EPS,
    exponential_primacy_decay,
    relative_power_scale,
)
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
    Parameters,
    RecallDataset,
    ResponsePolicyCreateFn,
)

__all__ = [
    "CMR",
    "make_factory",
]

class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        connections: Float[Array, " item_count item_count"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        response_policy_create_fn: ResponsePolicyCreateFn["CMR"] = PositionalStop,
    ) -> None:
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.semantic_scale = parameters["semantic_scale"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.msem = connections * self.semantic_scale
        self.response_policy = response_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.studied = jnp.zeros(self.item_count, dtype=bool)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """The learning rate for the MCF memory under its current state."""
        return self._mcf_learning_rate[self.study_index]

    def experience_item(self, item_index: Int_) -> "CMR":
        """Return the model after experiencing item with the specified index.

        Args:
            item_index: the index of the item to experience. 0-indexed.
        """
        item = self.items[item_index]
        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(item, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(learning_state, item, self.mcf_learning_rate),
            studied=self.studied.at[item_index].set(True),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> "CMR":
        """Returns model after simulating the specified study event.

        Args:
            choice: the index of the item to experience (1-indexed). 0 is ignored.
        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "CMR":
        """Returns model after transitioning from study to retrieval mode."""
        start_input = self.context.initial_state
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of item with the specified index.

        Args:
            choice: the index of the item to retrieve (0-indexed)
        """
        new_context = self.context.integrate(
            self.mfc.probe(self.items[item_index]),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(self.allow_repeated_recalls),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "CMR":
        """Return model after simulating the specified retrieval event.

        Args:
            choice: the index of the item to retrieve (1-indexed) or 0 to stop.
        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Compute item weights for the selected studied candidates."""
        candidates = candidates & self.studied
        base_support = self.mcf.probe(self.context.state) * candidates
        semantic_support = (
            lax.cond(
                self.recall_total == 0,
                lambda: jnp.ones_like(base_support),
                lambda: self.msem[self.recalls[self.recall_total - 1] - 1],
            )
            * candidates
        )
        scaled_base = relative_power_scale(base_support, sensitivity)
        scaled_semantic = relative_power_scale(semantic_support, self.semantic_scale)
        combined = scaled_base * scaled_semantic
        return (combined + FLOAT32_EPS) * candidates

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return probabilities in recorded order: STOP, then each local item."""
        return self.response_policy.outcome_probabilities(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Score the recorded STOP or item using the same distribution as simulation."""
        return self.outcome_probabilities()[choice]


def make_factory(
    mfc_create_fn: MemoryCreateFn,
    mcf_create_fn: MemoryCreateFn,
    context_create_fn: ContextCreateFn,
    response_policy_create_fn: ResponsePolicyCreateFn[CMR],
) -> type[MemorySearchModelFactory]:
    """Build multiplicative semantic CMR from vocabulary feature vectors.

    The factory uses `pres_itemnos` to put vocabulary IDs from `pres_itemids`
    in local item-code order. The model uses `choice_sensitivity`
    for temporal support and `semantic_scale` for semantic support. With
    positive `semantic_scale`, the strongest positive semantic match among
    available items has weight one. If `features` is `None`, all semantic
    connections are zero. After the first recall only the common support floor
    remains among available items.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Construct item-to-context memory.
    mcf_create_fn : MemoryCreateFn
        Construct context-to-item memory.
    context_create_fn : ContextCreateFn
        Construct temporal context for the trial.
    response_policy_create_fn : ResponsePolicyCreateFn[CMR]
        Assign probabilities to STOP and each item.

    Returns
    -------
    type[MemorySearchModelFactory]
        Instantiate with `(dataset, features)` and use `create_trial_model` to
        select the trial's connections.
    """

    class CMRModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.present_lists = jnp.asarray(_require_pres_itemids(dataset))
            self.max_list_length = np.max(dataset["listLength"]).item()
            vocabulary_ids = global_item_ids_by_code(
                dataset["pres_itemnos"], self.present_lists, self.max_list_length
            )
            self.trial_connections = build_trial_connections(vocabulary_ids, features)

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
                    response_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Parameters) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length, parameters, self.trial_connections[0]
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length, parameters, self.trial_connections[trial_index]
            )

    return CMRModelFactory
