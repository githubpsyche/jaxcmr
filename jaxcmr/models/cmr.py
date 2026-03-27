"""Context Maintenance and Retrieval (CMR) model of memory search.

Implements the core CMR architecture: study-phase context
integration, associative encoding via outer-product learning, and
retrieval via context-cued competition with a Luce choice rule.

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

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ):
        """Context Maintenance and Retrieval model of memory search.

        Parameters
        ----------
        list_length : int
            Number of items in the study list.
        parameters : Mapping[str, Float_]
            Model parameters including drift rates, learning rates, and
            sensitivity values.
        mfc_create_fn : MemoryCreateFn, optional
            Factory function for item-to-context memory (M_FC).
        mcf_create_fn : MemoryCreateFn, optional
            Factory function for context-to-item memory (M_CF).
        context_create_fn : ContextCreateFn, optional
            Factory function for temporal context representation.
        termination_policy_create_fn : TerminationPolicyCreateFn, optional
            Factory function for recall termination policy.

        """
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        # self.delay_drift_rate = parameters["delay_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
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
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """Learning rate for context-to-item memory at current study position.

        Returns
        -------
        Float[Array, ""]
            Position-dependent learning rate from primacy gradient.

        """
        return self._mcf_learning_rate[self.study_index]

    def experience_item(self, item_index: Int_) -> "CMR":
        """Simulate encoding of an item during study, updating context and memories.

        Parameters
        ----------
        item_index : Int_
            Index of the item to experience (0-indexed).

        Returns
        -------
        CMR
            Updated model state after encoding the item.

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
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> "CMR":
        """Simulate a study event.

        Parameters
        ----------
        choice : Int_
            Index of item to experience (1-indexed). A value of 0 is
            ignored and returns the model unchanged.

        Returns
        -------
        CMR
            Updated model state after the study event.

        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "CMR":
        """Transition from study to retrieval mode by reinstating start context.

        Returns
        -------
        CMR
            Model state ready to begin retrieval.

        """
        # delay_input = jnp.mean(self.mcf.state, axis=1)
        start_input = self.context.initial_state
        # start_context = self.context.integrate(
        #     delay_input, self.delay_drift_rate
        # ).integrate(start_input, self.start_drift_rate)
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Simulate retrieval of a specific item, updating context and records.

        Parameters
        ----------
        item_index : Int_
            Index of the item to retrieve (0-indexed).

        Returns
        -------
        CMR
            Updated model state after retrieval.

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
        """Simulate a retrieval event.

        Parameters
        ----------
        choice : Int_
            Index of item to retrieve (1-indexed), or 0 to terminate
            recall.

        Returns
        -------
        CMR
            Updated model state after the retrieval event.

        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations for all items from context-to-item memory.

        Returns
        -------
        Float[Array, " item_count"]
            Relative retrieval support for each item.

        """
        _activations = self.mcf.probe(self.context.state) * self.recallable
        return (power_scale(_activations, self.mcf_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Compute probability of terminating recall.

        Returns
        -------
        Float[Array, ""]
            Probability of stopping retrieval given current state.

        """
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Compute probability of retrieving a specific item, assuming recall continues.

        Parameters
        ----------
        item_index : Int_
            Index of the item (0-indexed).

        Returns
        -------
        Float[Array, ""]
            Probability of retrieving the specified item.

        """
        item_activations = self.activations()
        return item_activations[item_index] / jnp.sum(item_activations)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Compute probability of a specific retrieval outcome.

        Parameters
        ----------
        choice : Int_
            Index of item to retrieve (1-indexed), or 0 for recall
            termination.

        Returns
        -------
        Float[Array, ""]
            Probability of the specified outcome.

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
        """Compute probabilities for all possible retrieval outcomes.

        Returns
        -------
        Float[Array, " recall_outcomes"]
            Probability distribution over outcomes, where index 0 is
            termination and indices 1 to item_count are item recalls.

        """
        p_stop = self.stop_probability()
        item_activation = self.activations()
        item_activation_sum = jnp.sum(item_activation)
        return jnp.hstack(
            (
                p_stop,
                (
                    (1 - p_stop)
                    * item_activation
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
    """Create a CMR model factory with specified component functions.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Factory function for item-to-context memory (M_FC).
    mcf_create_fn : MemoryCreateFn
        Factory function for context-to-item memory (M_CF).
    context_create_fn : ContextCreateFn
        Factory function for temporal context representation.
    termination_policy_create_fn : TerminationPolicyCreateFn
        Factory function for recall termination policy.

    Returns
    -------
    Type[MemorySearchModelFactory]
        A factory class that creates CMR model instances.

    """

    class CMRModelFactory:
        """Factory for creating CMR model instances; sets `max_list_length` from dataset.

        Parameters
        ----------
        dataset : RecallDataset
            Dataset containing recall trials with list length information.
        features : Float[Array, " word_pool_items features_count"] or None
            Optional semantic feature vectors for items in the word pool.

        """

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
            """Create a CMR model instance with the given parameters.

            Parameters
            ----------
            parameters : Mapping[str, Float_]
                Model parameters.

            Returns
            -------
            MemorySearch
                Initialized CMR model instance.

            """
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            """Create a CMR model instance for a specific trial.

            Parameters
            ----------
            trial_index : Integer[Array, ""]
                Index of the trial (unused in base CMR).
            parameters : Mapping[str, Float_]
                Model parameters.

            Returns
            -------
            MemorySearch
                Initialized CMR model instance.

            """
            return self.model_create_fn(self.max_list_length, parameters)

    return CMRModelFactory
