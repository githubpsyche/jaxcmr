"""
Zero Pre-Experimental Positional CMR.

Items are encoded using positional representations rather than item representations.
This allows repeated items to have distinct contextual associations for each presentation.
When an item is re-presented, its pre-experimental connections are zeroed out before
learning the new position-context association.
"""

from typing import Mapping, Optional, Type

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.math import (
    exponential_primacy_decay,
    lb,
    power_scale,
)
from jaxcmr.typing import (
    Array,
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
    "CMR",
    "make_factory",
]

class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ):
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.mfc_sensitivity = parameters.get("mfc_choice_sensitivity", 1.0)
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        #! item representations on F now position representations
        self.positions = jnp.eye(self.item_count)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        #! We track studied item for each study position
        self.item_ids = jnp.arange(list_length)
        self.studied = jnp.zeros(list_length, dtype=int)
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        #! recallable is per-position, not per-item
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
        #! instead of probing and learning using item, we use the item's study position
        mfc_cue = self.positions[self.study_index]
        context_input = self.mfc.probe(mfc_cue)
        #! if this item has been studied before, zero out its pre-experimental connections
        intermediate_mfc = lax.cond(
            jnp.isin(item_index + 1, self.studied),
            lambda: self.mfc.zero_out(self.study_index),
            lambda: self.mfc,
        )
        intermediate_mcf = lax.cond(
            jnp.isin(item_index + 1, self.studied),
            lambda: self.mcf.zero_out(self.study_index + 1),
            lambda: self.mcf,
        )
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )
        return self.replace(
            context=new_context,
            #! associate using position cue instead of item, with zeroed-out memories
            mfc=intermediate_mfc.associate(
                mfc_cue, learning_state, self.mfc_learning_rate
            ),
            mcf=intermediate_mcf.associate(
                learning_state, mfc_cue, self.mcf_learning_rate
            ),
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
            item_index: the index of the item to retrieve (0-indexed)
        """
        #! We don't know which trace was recalled,
        #! so we use relative support from MCF to weight recall
        item_activation = self.position_activations() * (self.studied == item_index + 1)
        mfc_cue = power_scale(
            item_activation / jnp.sum(item_activation), self.mfc_sensitivity
        )
        new_context = self.context.integrate(
            self.mfc.probe(mfc_cue),
            self.recall_drift_rate,
        )
        #! find all study positions of the recalled item and set to not recallable
        recallable = lax.cond(
            self.allow_repeated_recalls,
            lambda: self.recallable,
            lambda: self.recallable * (self.studied != item_index + 1),
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=recallable,
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

    def position_activations(self) -> Float[Array, " list_length"]:
        """Returns relative support for retrieval of each study position given model state"""
        #! refactored to get position activations separately
        _activations = self.mcf.probe(self.context.state) * self.recallable
        return (power_scale(_activations, self.mcf_sensitivity) + lb) * self.recallable

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state"""
        #! reworked to pool position activations by item
        position_activations = self.position_activations()
        return lax.map(
            lambda i: jnp.sum(position_activations * (self.studied == i + 1)),
            self.item_ids,
        )

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state"""
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Assumes that some items are recallable, with at least the minimum recall probability.

        Args:
            item_index: the index of the item to retrieve.
        """
        #! Since item activations are potentially distributed across position activations,
        #! instead of indexing by item, we mask position activations by item then sum/normalize
        position_activations = self.position_activations()
        item_activation = jnp.sum(
            position_activations * (self.studied == item_index + 1)
        )
        return item_activation / jnp.sum(position_activations)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return probability of the specified retrieval event.

        Args:
            choice: the index of the item to retrieve (1-indexed) or 0 to stop.
        """
        p_stop = self.stop_probability()
        return lax.cond(
            choice == 0,
            lambda: p_stop,
            lambda: lax.cond(
                jnp.logical_or(p_stop == 1.0, ~self.recallable[choice - 1]),
                lambda: 0.0,
                lambda: (1 - p_stop) * self.item_probability(choice - 1),
            ),
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return the outcome probabilities of all recall events."""
        p_stop = self.stop_probability()
        item_activations = self.activations()
        item_activation_sum = jnp.sum(item_activations)
        return jnp.hstack(
            (
                p_stop,
                (
                    (1 - p_stop)
                    * item_activations
                    / lax.select(item_activation_sum == 0, 1.0, item_activation_sum)
                ),
            )
        )


def make_factory(
    mfc_create_fn: MemoryCreateFn,
    mcf_create_fn: MemoryCreateFn,
    context_create_fn: ContextCreateFn,
    termination_policy_create_fn: TerminationPolicyCreateFn,
) -> Type[MemorySearchModelFactory]:
    class CMRModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ):
            self.max_list_length = np.max(dataset["listLength"]).item()

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
            ) -> MemorySearch:
                return CMR(
                    list_length,
                    parameters,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    termination_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Mapping[str, Float_]) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

    return CMRModelFactory
