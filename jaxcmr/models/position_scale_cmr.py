from typing import Mapping, Optional

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.models.context import TemporalContext
from jaxcmr.models.instance_memory import InstanceMemory
from jaxcmr.models.linear_memory import LinearMemory
from jaxcmr.math import exponential_primacy_decay, exponential_stop_probability, power_scale, lb
from jaxcmr.typing import (
    Array,
    Context,
    Float,
    Float_,
    Int_,
    Integer,
    Memory,
    MemorySearch,
    RecallDataset,
)


class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        mfc: Memory,
        mcf: Memory,
        context: Context,
    ):
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.shared_support = parameters["shared_support"]
        self.item_support = parameters["item_support"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.stop_probability_scale = parameters["stop_probability_scale"]
        self.stop_probability_growth = parameters["stop_probability_growth"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.mfc_sensitivity = parameters.get("mfc_choice_sensitivity", 1.0)
        self.item_count = list_length
        #! item representations on F now position representations 
        self.positions = jnp.eye(list_length)
        self._stop_probability = exponential_stop_probability(
            self.stop_probability_scale,
            self.stop_probability_growth,
            jnp.arange(list_length),
        )
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        #! We track studied item for each study position
        self.item_ids = jnp.arange(list_length)
        self.studied = jnp.zeros(list_length, dtype=int)
        self.context = context
        self.mfc = mfc
        self.mcf = mcf
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
        mfc_cue = self.positions[self.study_index] # item = self.items[item_index]
        context_input = self.mfc.probe(mfc_cue)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(mfc_cue, new_context.state, self.mfc_learning_rate),
            mcf=self.mcf.associate(new_context.state, mfc_cue, self.mcf_learning_rate),
            #! also update recallable at the study position instead of item_index
            recallable=self.recallable.at[self.study_index].set(True),
            #! and track each item's study position(s)
            studied=self.studied.at[self.study_index].set(item_index+1),
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
        mfc_cue = power_scale(item_activation / jnp.sum(item_activation), self.mfc_sensitivity)
        new_context = self.context.integrate(
            self.mfc.probe(mfc_cue),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            #! find all study positions of the recalled item and set to not recallable
            # recallable=self.recallable.at[item_index].set(False),
            recallable= self.recallable * (self.studied != item_index + 1),
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
        position_activations = self.mcf.probe(self.context.state) + lb
        return position_activations * self.recallable  # mask recalled study positions
    
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
        total_recallable = jnp.sum(self.recallable)
        return lax.cond(
            total_recallable == 0,
            true_fun=lambda: 1.0,
            false_fun=lambda: lax.cond(
                self.is_active,
                true_fun=lambda: jnp.minimum(
                    1.0 - (lb * total_recallable),
                    self._stop_probability[self.recall_total],
                ),
                false_fun=lambda: 1.0,
            ),
        )

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Assumes that some items are recallable, with at least the minimum recall probability.

        Args:
            item_index: the index of the item to retrieve.
        """
        #! Since item activations are potentially distributed across position activations,
        #! instead of indexing by item, we mask position activations by item then sum/normalize
        p_continue = 1 - self.stop_probability()
        position_activations = self.position_activations()
        item_activation = jnp.sum(
            position_activations * (self.studied == item_index + 1)
        )
        return p_continue * (item_activation / jnp.sum(position_activations))

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
                p_stop == 1.0,
                lambda: 0.0,
                lambda: self.item_probability(choice - 1),
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


def BaseCMR(list_length: int, parameters: Mapping[str, Float_]) -> CMR:
    """Creates a regular CMR model with linear associative $M^{FC}$ and $M^{CF}$ memories."""
    context = TemporalContext.init(list_length)
    mfc = LinearMemory.init_mfc(
        list_length,
        context.size,
        parameters["learning_rate"],
        parameters.get("mfc_choice_sensitivity", 1.0),
    )
    mcf = LinearMemory.init_mcf(
        list_length,
        context.size,
        parameters["item_support"],
        parameters["shared_support"],
        parameters["choice_sensitivity"],
    )
    return CMR(list_length, parameters, mfc, mcf, context)


def InstanceCMR(list_length: int, parameters: Mapping[str, Float_]) -> CMR:
    """
    Creates InstanceCMR model with instance-based $M^{FC}$ and $M^{CF}$ memories.

    Equivalent to the original CMR model when `mcf_trace_sensitivity` is set to 1.0.
    Usually slower than the linear version, but often more interpretable and flexible.
    """
    context = TemporalContext.init(list_length)
    mfc = InstanceMemory.init_mfc(
        list_length,
        context.size,
        list_length,
        parameters["learning_rate"],
        parameters.get("mfc_choice_sensitivity", 1.0),
        parameters.get("mfc_trace_sensitivity", 1.0),
    )
    mcf = InstanceMemory.init_mcf(
        list_length,
        context.size,
        list_length,
        parameters["item_support"],
        parameters["shared_support"],
        parameters["choice_sensitivity"],
        parameters["mcf_trace_sensitivity"],
    )
    return CMR(list_length, parameters, mfc, mcf, context)


def MixedCMR(list_length: int, parameters: Mapping[str, Float_]) -> CMR:
    """
    Creates MixedCMR model with linear $M^{FC}$ and instance-based $M^{CF}$ memories.

    Equivalent to InstanceCMR but faster feature-to-context memory.
    """
    context = TemporalContext.init(list_length)
    mfc = LinearMemory.init_mfc(
        list_length,
        context.size,
        parameters["learning_rate"],
        parameters.get("mfc_choice_sensitivity", 1.0),
    )
    mcf = InstanceMemory.init_mcf(
        list_length,
        context.size,
        list_length,
        parameters["item_support"],
        parameters["shared_support"],
        parameters["choice_sensitivity"],
        parameters["mcf_trace_sensitivity"],
    )
    return CMR(list_length, parameters, mfc, mcf, context)


class BaseCMRFactory:
    def __init__(
        self,
        dataset: RecallDataset,
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()

    def create_model(
        self,
        trial_index: Int_,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return BaseCMR(self.max_list_length, parameters)


class InstanceCMRFactory:
    def __init__(
        self,
        dataset: RecallDataset,
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()

    def create_model(
        self,
        trial_index: Int_,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return InstanceCMR(self.max_list_length, parameters)


class MixedCMRFactory:
    def __init__(
        self,
        dataset: RecallDataset,
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()

    def create_model(
        self,
        trial_index: Int_,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return MixedCMR(self.max_list_length, parameters)
