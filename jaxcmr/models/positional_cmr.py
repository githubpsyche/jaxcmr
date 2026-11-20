"""Positional CMR learns an association at each study position. On a list
`[A, B, A]`, context can cue both the first and third positions. Their
supports add to give the support for recalling A.

A recall identifies the item. To reinstate context, the model weights
that item's positions by their current support. If repeated recalls are
disabled, recalling A makes both of its positions unavailable.
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
    """A trial model with one associative cue for each study position.

    Parameters
    ----------
    list_length : int
        Capacity for study positions and local item codes.
    parameters : Parameters
        The temporal learning, drift, primacy and recall parameters used by
        ordinary CMR. `choice_sensitivity` weights positions for item choice.
        `mfc_sensitivity` weights the recalled item's positions for context
        reinstatement.
    mfc_create_fn : MemoryCreateFn, optional
        Construct position-to-context memory.
    mcf_create_fn : MemoryCreateFn, optional
        Construct context-to-position memory.
    context_create_fn : ContextCreateFn, optional
        Construct temporal context.
    response_policy_create_fn : ResponsePolicyCreateFn[CMR], optional
        Choose STOP and the next item. The default schedules STOP by recall count.
    """

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
        #! We associate with current context state instead of new_context in this implementation
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(mfc_cue, learning_state, self.mfc_learning_rate),  #! updated
            mcf=self.mcf.associate(learning_state, mfc_cue, self.mcf_learning_rate),  #! updated
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
        #! mfc_sensitivity scales the sharpness of this weighting; 
        #! higher values lead to more deterministic recalls focused on the strongest trace
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
    """Build a factory for trials that associate temporal context with study positions.

    The model receives within-list item codes during study. Successive
    presentations occupy successive positions, including repeat presentations
    of the same item.

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
        Instantiate with `(dataset, features=None)`. `create_trial_model` returns
        a fresh positional model sized to the dataset's maximum list length.
    """

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
