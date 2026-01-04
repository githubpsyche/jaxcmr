"""
Blend Positional CMR.

Maintains dual context streams: one item-dependent (CMR-style) and one
position-dependent (positional CMR-style). A blend_weight parameter controls
the relative contribution of each stream to recall competition.

- blend_weight = 0: pure positional CMR behavior
- blend_weight = 1: pure item-based CMR behavior
- 0 < blend_weight < 1: hybrid behavior

This allows modeling intermediate levels of context sharing across repeated
item presentations, addressing the empirical finding that erroneous transitions
to neighbors of second occurrences are above chance (unlike pure positional CMR)
but below what pure CMR predicts.
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
    """Blend Positional CMR with dual item/position context streams."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ):
        # Shared parameters
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

        # Blend parameter: 0 = pure positional, 1 = pure item-based
        self.blend_weight = parameters["blend_weight"]

        self.item_count = list_length

        # Dual representations: items and positions
        self.items = jnp.eye(list_length)
        self.positions = jnp.eye(list_length)

        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )

        # Track studied item for each study position (positional CMR style)
        self.item_ids = jnp.arange(list_length)
        self.studied = jnp.zeros(list_length, dtype=int)

        # Position-based context stream (positional CMR style)
        self.position_context = context_create_fn(list_length)
        self.position_mfc = mfc_create_fn(
            list_length, parameters, self.position_context
        )
        self.position_mcf = mcf_create_fn(
            list_length, parameters, self.position_context
        )

        # Item-based context stream (regular CMR style)
        self.item_context = context_create_fn(list_length)
        self.item_mfc = mfc_create_fn(list_length, parameters, self.item_context)
        self.item_mcf = mcf_create_fn(list_length, parameters, self.item_context)

        self.termination_policy = termination_policy_create_fn(list_length, parameters)
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

        Updates both position-based and item-based context streams.

        Args:
            item_index: the index of the item to experience. 0-indexed.
        """
        # Position-based stream: probe and learn using position
        position_cue = self.positions[self.study_index]
        position_context_input = self.position_mfc.probe(position_cue)
        new_position_context = self.position_context.integrate(
            position_context_input, self.encoding_drift_rate
        )
        position_learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_position_context.state,
            lambda: self.position_context.state,
        )

        # Item-based stream: probe and learn using item
        item_cue = self.items[item_index]
        item_context_input = self.item_mfc.probe(item_cue)
        new_item_context = self.item_context.integrate(
            item_context_input, self.encoding_drift_rate
        )
        item_learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_item_context.state,
            lambda: self.item_context.state,
        )

        return self.replace(
            # Update position-based stream
            position_context=new_position_context,
            position_mfc=self.position_mfc.associate(
                position_cue, position_learning_state, self.mfc_learning_rate
            ),
            position_mcf=self.position_mcf.associate(
                position_learning_state, position_cue, self.mcf_learning_rate
            ),
            # Update item-based stream
            item_context=new_item_context,
            item_mfc=self.item_mfc.associate(
                item_cue, item_learning_state, self.mfc_learning_rate
            ),
            item_mcf=self.item_mcf.associate(
                item_learning_state, item_cue, self.mcf_learning_rate
            ),
            # Recallability at position level
            recallable=self.recallable.at[self.study_index].set(True),
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
        # Both contexts drift toward start-of-list
        new_position_context = self.position_context.integrate(
            self.position_context.initial_state, self.start_drift_rate
        )
        new_item_context = self.item_context.integrate(
            self.item_context.initial_state, self.start_drift_rate
        )
        return self.replace(
            position_context=new_position_context,
            item_context=new_item_context,
        )

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of item with the specified index.

        Updates both context streams based on the retrieved item.

        Args:
            item_index: the index of the item to retrieve (0-indexed)
        """
        # Position-based stream update (like positional CMR)
        # Weight positions by relative support from MCF
        position_activation = self._position_stream_activations() * (
            self.studied == item_index + 1
        )
        position_mfc_cue = power_scale(
            position_activation / jnp.sum(position_activation), self.mfc_sensitivity
        )
        new_position_context = self.position_context.integrate(
            self.position_mfc.probe(position_mfc_cue),
            self.recall_drift_rate,
        )

        # Item-based stream update (like regular CMR)
        item_cue = self.items[item_index]
        new_item_context = self.item_context.integrate(
            self.item_mfc.probe(item_cue),
            self.recall_drift_rate,
        )

        return self.replace(
            position_context=new_position_context,
            item_context=new_item_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            # Recallability at position level
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

    def _position_stream_activations(self) -> Float[Array, " list_length"]:
        """Returns scaled position activations from position-based context stream.

        Used by retrieve_item() to weight which position trace was retrieved.
        """
        _activations = (
            self.position_mcf.probe(self.position_context.state) * self.recallable
        )
        return (power_scale(_activations, self.mcf_sensitivity) + lb) * self.recallable

    def _raw_position_stream(self) -> Float[Array, " list_length"]:
        """Returns raw (unscaled) position activations from position-based stream."""
        return self.position_mcf.probe(self.position_context.state) * self.recallable

    def _raw_item_stream(self) -> Float[Array, " list_length"]:
        """Returns raw (unscaled) position activations from item-based stream.

        The item MCF maps context -> items, so we redistribute item activations
        back to positions based on which item was studied where.
        """
        # Get raw item activations from item-based MCF
        item_activations = self.item_mcf.probe(self.item_context.state)

        # Map item activations to positions based on studied mapping
        # For each position, get the activation of the item studied there
        position_activations = lax.map(
            lambda pos: lax.cond(
                self.studied[pos] > 0,
                lambda: item_activations[self.studied[pos] - 1],
                lambda: 0.0,
            ),
            jnp.arange(self.item_count),
        )
        return position_activations * self.recallable

    def position_activations(self) -> Float[Array, " list_length"]:
        """Returns blended position activations from both context streams.

        Uses normalize-then-blend-then-scale approach:
        1. Normalize each stream to probability distribution
        2. Blend with mixture weights (blend_weight controls contribution)
        3. Apply power scaling once to combined result
        """
        raw_pos = self._raw_position_stream()
        raw_item = self._raw_item_stream()

        # Normalize each stream to probability distribution
        pos_prob = raw_pos / (jnp.sum(raw_pos) + lb)
        item_prob = raw_item / (jnp.sum(raw_item) + lb)

        # Blend distributions (mixture model: blend_weight = prob of using item stream)
        blended = (1 - self.blend_weight) * pos_prob + self.blend_weight * item_prob

        # Single winner-take-all stage on combined distribution
        return (power_scale(blended, self.mcf_sensitivity) + lb) * self.recallable

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state."""
        position_activations = self.position_activations()
        return lax.map(
            lambda i: jnp.sum(position_activations * (self.studied == i + 1)),
            self.item_ids,
        )

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state."""
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Args:
            item_index: the index of the item to retrieve.
        """
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
                jnp.logical_or(
                    p_stop == 1.0,
                    jnp.all(self.recallable * (self.studied == choice) == 0),
                ),
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
