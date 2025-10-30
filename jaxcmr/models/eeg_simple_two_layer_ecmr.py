"""
CMR: Context Maintenance and Retrieval model of memory search.
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
    Bool,
    Int_,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
    TerminationPolicyCreateFn,
)


class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_emotional: Bool[Array, " study_events"],
        lpp_centered: Float[Array, " study_events"],
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
        self.emotion_scale = parameters["emotion_scale"]
        self.modulate_emotion_by_primacy = parameters["modulate_emotion_by_primacy"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)

        self.primacy = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.is_emotional = jnp.array(is_emotional, dtype=jnp.float32)

        # Centered LPP provided by factory so slope/threshold shape relative boosts.
        lpp_slope = parameters["lpp_slope"]
        lpp_threshold = parameters["lpp_threshold"]
        self.emotion_modulation = (
            lpp_slope * (lpp_centered - lpp_threshold) * self.is_emotional
        )
        self.emotional_mcf = self.is_emotional
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

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
        base_emotion_strength = jnp.maximum(
            0.0, self.emotion_scale + self.emotion_modulation[self.study_index]
        )
        emcf_lr = lax.cond(
            self.modulate_emotion_by_primacy,
            lambda: self.primacy[self.study_index] * base_emotion_strength,
            lambda: base_emotion_strength,
        )
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(item, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(learning_state, item, self.primacy[self.study_index]),
            emotional_mcf=self.emotional_mcf.at[item_index].multiply(emcf_lr),
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

    def activations(self):
        """Returns relative support for retrieval of each item given model state"""
        _activations = self.mcf.probe(self.context.state) * self.recallable
        
        last_emotional = lax.cond(
            self.recall_total > 0,
            lambda: self.is_emotional[self.recalls[self.recall_total - 1] - 1] > 0,
            lambda: False,
        )
        emotion_support = last_emotional * self.emotional_mcf

        merged_support = power_scale(
            _activations + emotion_support,
            self.mcf_sensitivity,
        )
        return (merged_support + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state"""
        return self.termination_policy.stop_probability(self)

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
            self.present_lists = np.array(dataset["pres_itemids"])
            self.max_list_length = np.max(dataset["listLength"]).item()

            # 0 for neutral study events, 1 for emotional study events
            self.trial_emotions = (2 - dataset["condition"]).astype(bool)
            lpp_raw = jnp.array(dataset["EarlyLPP"], dtype=jnp.float32)
            valid_lpp = jnp.where(self.trial_emotions, lpp_raw, 0.0)
            emotional_count = jnp.maximum(
                jnp.sum(self.trial_emotions, axis=1, keepdims=True),
                1.0,
            )
            emotional_mean = jnp.sum(valid_lpp, axis=1, keepdims=True) / emotional_count
            centered_lpp = jnp.where(
                self.trial_emotions, valid_lpp - emotional_mean, 0.0
            )
            self.lpp_centered = centered_lpp

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_emotional: Bool[Array, " study_events"],
                lpp_centered: Float[Array, " study_events"],
            ) -> MemorySearch:
                return CMR(
                    list_length,
                    parameters,
                    is_emotional,
                    lpp_centered,
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
                self.trial_emotions[0],
                self.lpp_centered[0],
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_emotions[trial_index],
                self.lpp_centered[trial_index],
            )

    return CMRModelFactory
