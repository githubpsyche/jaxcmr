"""Positional CMR with trace reinforcement and repetition learning allocation.

Items are encoded using positional representations rather than item
representations. This allows repeated items to have distinct
contextual associations for each presentation. Recallability is
tracked at the position level (not item level).

Reinforcement uses the saved context of the first occurrence, independently
of ongoing encoding context. MCF and MFC gains can be separate or shared.
Repetition encoding costs reduce only the new occurrence's learned links.
``repetition_allocation`` redirects a fraction of each current learning
increment to the earlier trace, including MCF's actual primacy weighting.
All additions default to zero, preserving the original MCF-only model.

"""



import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.response import (
    PositionalStop,
    pool_position_weights,
    relative_choice_weights,
)
from jaxcmr.math import (
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
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.mfc_sensitivity = parameters["mfc_sensitivity"]
        shared = parameters.get("first_pres_reinforcement", 0.0)
        self.mcf_first_pres_reinforcement = shared + parameters.get(
            "mcf_first_pres_reinforcement", 0.0
        )
        self.mfc_first_pres_reinforcement = shared + parameters.get(
            "mfc_first_pres_reinforcement", 0.0
        )
        encoding_cost = parameters.get("repetition_encoding_cost", 0.0)
        self.mcf_repetition_encoding_cost = encoding_cost + parameters.get(
            "mcf_repetition_encoding_cost", 0.0
        )
        self.mfc_repetition_encoding_cost = encoding_cost + parameters.get(
            "mfc_repetition_encoding_cost", 0.0
        )
        self.repetition_allocation = parameters.get("repetition_allocation", 0.0)
        self.trace_reinforcement_after_context_update = parameters.get(
            "trace_reinforcement_after_context_update", False
        )
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        #! item representations on F now position representations
        self.positions = jnp.eye(list_length)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        #! We track studied item for each study position
        self.item_ids = jnp.arange(list_length)
        self.studied = jnp.zeros(list_length, dtype=int)
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.trace_contexts = jnp.zeros((list_length, self.context.size))
        self.response_policy = response_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(list_length, dtype=int)
        self.recallable = jnp.zeros(list_length, dtype=bool)
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
        #! instead of probing and learning using item, we use the item's study position
        mfc_cue = self.positions[self.study_index]
        context_input = self.mfc.probe(mfc_cue)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )
        trace_state = lax.cond(
            self.trace_reinforcement_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )

        #! if studied item has been studied before, we reinforce the first trace context
        item_studied = self.studied == item_index + 1
        already_studied = jnp.any(item_studied)
        first_item_studied = self.item_ids == jnp.argmax(item_studied)
        associated_context = jnp.dot(first_item_studied, self.trace_contexts)
        new_mfc = lax.cond(
            already_studied,
            lambda: self.mfc.associate(
                first_item_studied,
                associated_context,
                self.mfc_first_pres_reinforcement
                + self.repetition_allocation * self.mfc_learning_rate,
            ),
            lambda: self.mfc,
        )
        new_mcf = lax.cond(
            already_studied,
            lambda: self.mcf.associate(
                associated_context,
                first_item_studied,
                self.mcf_first_pres_reinforcement
                + self.repetition_allocation * self.mcf_learning_rate,
            ),
            lambda: self.mcf,
        )
        mfc_learning_scale = jnp.where(
            already_studied,
            (1 - self.mfc_repetition_encoding_cost) * (1 - self.repetition_allocation),
            1.0,
        )
        mcf_learning_scale = jnp.where(
            already_studied,
            (1 - self.mcf_repetition_encoding_cost) * (1 - self.repetition_allocation),
            1.0,
        )

        #! We associate with current context state instead of new_context in this implementation
        return self.replace(
            context=new_context,
            mfc=new_mfc.associate(
                mfc_cue, learning_state, self.mfc_learning_rate * mfc_learning_scale
            ),  #! updated
            mcf=new_mcf.associate(
                learning_state, mfc_cue, self.mcf_learning_rate * mcf_learning_scale
            ),  #! updated
            trace_contexts=self.trace_contexts.at[self.study_index].set(trace_state),
            #! update recallable at the study position instead of item_index
            recallable=self.recallable.at[self.study_index].set(True),
            #! track each item's study position(s)
            studied=self.studied.at[self.study_index].set(item_index + 1),
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
        #! We don't know which trace was recalled,
        #! so we use relative support from MCF to weight recall
        item_activation = self.position_activations() * (self.studied == item_index + 1)
        mfc_cue = relative_power_scale(
            item_activation / jnp.sum(item_activation), self.mfc_sensitivity
        )
        new_context = self.context.integrate(
            self.mfc.probe(mfc_cue),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            #! Conditionally toggle recallability based on allow_repeated_recalls
            recallable=lax.cond(
                self.allow_repeated_recalls,
                lambda: self.recallable,
                lambda: self.recallable * (self.studied != item_index + 1),
            ),
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

    def position_weights(
        self, candidates: Bool[Array, " positions"], sensitivity: Float_
    ) -> Float[Array, " positions"]:
        """Weight the selected study positions before combining occurrences of an item."""
        return relative_choice_weights(
            self.mcf.probe(self.context.state),
            candidates & (self.studied > 0),
            sensitivity,
        )

    def position_activations(self) -> Float[Array, " positions"]:
        """Return the available position weights at the model's choice sensitivity."""
        return self.position_weights(self.recallable, self.mcf_sensitivity)

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Sum the selected occurrence weights for each recorded item."""
        return pool_position_weights(
            self.position_weights(candidates, sensitivity), self.studied
        )

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
    class CMRModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.max_list_length = np.max(dataset["listLength"]).item()

            def model_create_fn(
                list_length: int,
                parameters: Parameters,
            ) -> MemorySearch:
                return CMR(
                    list_length,
                    parameters,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    response_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Parameters) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

    return CMRModelFactory
