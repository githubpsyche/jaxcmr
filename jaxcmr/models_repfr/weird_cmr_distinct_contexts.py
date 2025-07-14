"""
CMR except each studied item is associated with the current context state instead of the updated context state after each experience. In other words, the studied item's associated contextual features only become associated in memory with the *next* studied item, not the current one.

This apparently a more common specification of CMR than the alternative variant where studied items are associated with the state of context after update. We've found that both variants have similar performance because they both achieve similar similarity structure between the context states associated with each studied item. However, there are nonetheless useful differences between the variants that can make them differently useful as starting points for further model development.

For example, in the case of intra-list serial repetitionn, if an item gets associated with the current instead of the updated context state, this reduces the overlap in contextual states between two presentations of the same item, which can reduce associative interference between the two presentations, making it easier to retrieve details of one presentation without interference from the other. However, this is only relevant when it's possible to selectively reinstate contexts associated with specific presentations even when items are repeated -- something actually still impossible under the specification provided here. 

In contexts like these, this specification can useful even thought it doesn't add new behavior on its own. It may similarly be possible for the original CMR specification to be a useful starting point in certain domains. 
"""

from typing import Mapping, Optional

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.math import (
    exponential_primacy_decay,
    exponential_stop_probability,
    lb,
    power_scale,
)
from jaxcmr.models.context import TemporalContext
from jaxcmr.models.instance_memory import InstanceMemory
from jaxcmr.models.linear_memory import LinearMemory
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
        self.allow_repeated_recalls = parameters.get("allow_repeated_recalls", False)
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)
        self._stop_probability = exponential_stop_probability(
            self.stop_probability_scale,
            self.stop_probability_growth,
            jnp.arange(self.item_count),
        )
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = context
        self.mfc = mfc
        self.mcf = mcf
        self.recalls = jnp.zeros(self.item_count, dtype=int)
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
        #! We associate with current context state instead of new_context in this implementation
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(
                item, self.context.state, self.mfc_learning_rate
            ),  #! updated
            mcf=self.mcf.associate(
                self.context.state, item, self.mcf_learning_rate
            ),  #! updated
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

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state"""
        _activations = self.mcf.probe(self.context.state) * self.recallable
        return (power_scale(_activations, self.mcf_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state"""
        total_recallable = jnp.sum(self.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~self.is_active),
            true_fun=lambda: 1.0,
            false_fun=lambda: jnp.minimum(
                1.0 - (lb * total_recallable),
                self._stop_probability[self.recall_total],
            ),
        )

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Assumes that some items are recallable, with at least the minimum recall probability.

        Args:
            item_index: the index of the item to retrieve.
        """
        item_activations = self.activations()
        return item_activations[item_index] / jnp.sum(item_activations)

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


def BaseCMR(list_length: int, parameters: Mapping[str, Float_]) -> CMR:
    """Creates a regular CMR model with linear associative $M^{FC}$ and $M^{CF}$ memories."""
    context = TemporalContext.init(list_length)
    mfc = LinearMemory.init_mfc(
        list_length,
        context.size,
        parameters["learning_rate"],
    )
    mcf = LinearMemory.init_mcf(
        list_length,
        context.size,
        parameters["item_support"],
        parameters["shared_support"],
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
        parameters.get("mfc_trace_sensitivity", 1.0),
    )
    mcf = InstanceMemory.init_mcf(
        list_length,
        context.size,
        list_length,
        parameters["item_support"],
        parameters["shared_support"],
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
    )
    mcf = InstanceMemory.init_mcf(
        list_length,
        context.size,
        list_length,
        parameters["item_support"],
        parameters["shared_support"],
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
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        return BaseCMR(self.max_list_length, parameters)

    def create_trial_model(
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
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        return InstanceCMR(self.max_list_length, parameters)

    def create_trial_model(
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
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        return MixedCMR(self.max_list_length, parameters)

    def create_trial_model(
        self,
        trial_index: Int_,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return MixedCMR(self.max_list_length, parameters)
