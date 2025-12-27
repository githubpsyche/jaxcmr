"""Plain attentional-strength memory search model."""

from typing import Mapping, Optional, Type

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from jaxcmr.components.termination import PositionalTermination
from jaxcmr.math import exponential_primacy_decay, lb, power_scale
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
    TerminationPolicyCreateFn,
)


class StrengthSearch(Pytree):
    """Represents a strength-based memory search without retrieved context."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_emotional: Bool[Array, " study_events"],
        lpp_centered: Float[Array, " study_events"],
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ):
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.choice_sensitivity = parameters["choice_sensitivity"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.modulate_emotion_by_primacy = parameters["modulate_emotion_by_primacy"]

        lpp_slope = parameters["lpp_slope"]
        lpp_threshold = parameters["lpp_threshold"]
        self.emotion_modulation = lpp_slope * (lpp_centered - lpp_threshold) * is_emotional

        self.item_count = list_length
        self.primacy = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.strengths = jnp.zeros(self.item_count, dtype=jnp.float32)
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    def experience_item(self, item_index: Int_) -> "StrengthSearch":
        """Returns model after encoding the specified item.

        Args:
            item_index: Index of the studied item (0-indexed).
        """
        primacy = self.primacy[self.study_index]
        emotion_modulation = self.emotion_modulation[item_index]
        total_strength = lax.cond(
            self.modulate_emotion_by_primacy,
            lambda: primacy * jnp.maximum(1, 1 + emotion_modulation),
            lambda: primacy + jnp.maximum(-primacy, emotion_modulation),
        )

        return self.replace(
            strengths=self.strengths.at[item_index].set(total_strength),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> "StrengthSearch":
        """Returns model after simulating the specified study event.

        Args:
            choice: Index of the studied item (1-indexed). Zero is ignored.
        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "StrengthSearch":
        """Returns model after preparing for retrieval."""
        return self

    def retrieve_item(self, item_index: Int_) -> "StrengthSearch":
        """Returns model after recalling the specified item.

        Args:
            item_index: Index of the recalled item (0-indexed).
        """
        return self.replace(
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(self.allow_repeated_recalls),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "StrengthSearch":
        """Returns model after simulating the specified retrieval event.

        Args:
            choice: Index of the recalled item (1-indexed) or zero to stop.
        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for recalling each item."""
        _activations = self.strengths * self.recallable
        return (power_scale(_activations, self.choice_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state."""
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Returns probability of recalling the specified item.

        Args:
            item_index: Index of the recalled item (0-indexed).
        """
        item_activations = self.activations()
        return item_activations[item_index] / jnp.sum(item_activations)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Returns probability of the specified retrieval outcome.

        Args:
            choice: Recall choice (1-indexed) or zero to stop.
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
        """Returns probabilities for stop and item recall choices."""
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
    mfc_create_fn,
    mcf_create_fn,
    context_create_fn,
    termination_policy_create_fn: TerminationPolicyCreateFn,
) -> Type[MemorySearchModelFactory]:
    """Returns a factory that builds strength-based memory search models.

    Args:
        mfc_create_fn: Unused associative-memory factory (ignored).
        mcf_create_fn: Unused associative-memory factory (ignored).
        context_create_fn: Unused context factory (ignored).
        termination_policy_create_fn: Factory that creates termination policies.
    """
    del mfc_create_fn, mcf_create_fn, context_create_fn

    class StrengthModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ):
            self.max_list_length = np.max(dataset["listLength"]).item()

            self.trial_emotions = (2 - dataset["condition"]).astype(bool)
            lpp_raw = jnp.array(dataset["EarlyLPP"], dtype=jnp.float32)
            valid_lpp = jnp.where(self.trial_emotions, lpp_raw, 0.0)
            emotional_count = jnp.maximum(
                jnp.sum(self.trial_emotions, axis=1, keepdims=True),
                1.0,
            )
            emotional_mean = jnp.sum(valid_lpp, axis=1, keepdims=True) / emotional_count
            self.lpp_centered = jnp.where(self.trial_emotions, valid_lpp - emotional_mean, 0.0)

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_emotional: Bool[Array, " study_events"],
                lpp_centered: Float[Array, " study_events"],
            ) -> MemorySearch:
                return StrengthSearch(
                    list_length,
                    parameters,
                    is_emotional,
                    lpp_centered,
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

    return StrengthModelFactory
