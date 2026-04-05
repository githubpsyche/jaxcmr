"""Blend Positional CMR.

Implements a single-context bridge between Standard CMR and Positional CMR.
The ``blend_weight`` parameter controls the probability that recall and
reinstatement proceed through the item-mediated route rather than the
episode/position-mediated route.

- ``blend_weight = 0``: pure positional CMR behavior
- ``blend_weight = 1``: pure item-based CMR behavior
- ``0 < blend_weight < 1``: route-mixture behavior

"""

from typing import Mapping, Optional, Type

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.math import exponential_primacy_decay, lb, power_scale
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


class _TerminationProxy(Pytree):
    """Minimal model-like object for route-specific termination queries."""

    def __init__(self, recallable, recall_total, is_active, context, mcf, studied):
        self.recallable = recallable
        self.recall_total = recall_total
        self.is_active = is_active
        self.context = context
        self.mcf = mcf
        self.studied = studied


class CMR(Pytree):
    """Single-context route-mixture bridge between CMR and Positional CMR."""

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
        self.mfc_sensitivity = parameters["mfc_sensitivity"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.blend_weight = parameters["blend_weight"]

        self.item_count = list_length
        self.items = jnp.eye(list_length)
        self.positions = jnp.eye(list_length)
        self.item_ids = jnp.arange(list_length)

        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )

        self.context = context_create_fn(list_length)

        # Standard-CMR item route.
        self.item_mfc = mfc_create_fn(list_length, parameters, self.context)
        self.item_mcf = mcf_create_fn(list_length, parameters, self.context)

        # Positional-CMR occurrence route.
        self.position_mfc = mfc_create_fn(list_length, parameters, self.context)
        self.position_mcf = mcf_create_fn(list_length, parameters, self.context)

        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(list_length, dtype=int)

        # Item-route bookkeeping.
        self.item_studied = jnp.zeros(list_length, dtype=bool)
        self.item_recallable = jnp.zeros(list_length, dtype=bool)

        # Position-route bookkeeping.
        self.studied = jnp.zeros(list_length, dtype=int)
        self.position_recallable = jnp.zeros(list_length, dtype=bool)

        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """Learning rate for context-to-feature memory at current study position."""
        return self._mcf_learning_rate[self.study_index]

    @property
    def recallable(self) -> Float[Array, " list_length"]:
        """Compatibility alias used by some tooling."""
        return self.position_recallable

    def _normalize(self, values: Float[Array, " n"]) -> Float[Array, " n"]:
        total = jnp.sum(values)
        return values / lax.select(total == 0, 1.0, total)

    def _item_mask(self, item_index: Int_) -> Float[Array, " list_length"]:
        return self.studied == (item_index + 1)

    def _route_stop_probability(
        self,
        recallable: Float[Array, " list_length"],
        mcf,
        studied: Float[Array, " list_length"],
    ) -> Float[Array, ""]:
        proxy = _TerminationProxy(
            recallable=recallable,
            recall_total=self.recall_total,
            is_active=self.is_active,
            context=self.context,
            mcf=mcf,
            studied=studied,
        )
        return self.termination_policy.stop_probability(proxy)

    def _item_route_activations(self) -> Float[Array, " item_count"]:
        raw = self.item_mcf.probe(self.context.state) * self.item_recallable
        return (power_scale(raw, self.mcf_sensitivity) + lb) * self.item_recallable

    def _position_route_position_activations(self) -> Float[Array, " list_length"]:
        raw = self.position_mcf.probe(self.context.state) * self.position_recallable
        return (power_scale(raw, self.mcf_sensitivity) + lb) * self.position_recallable

    def _position_route_item_activations(self) -> Float[Array, " item_count"]:
        position_activations = self._position_route_position_activations()
        return lax.map(
            lambda i: jnp.sum(position_activations * (self.studied == (i + 1))),
            self.item_ids,
        )

    def _item_route_outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        p_stop = self._route_stop_probability(
            self.item_recallable,
            self.item_mcf,
            self.item_studied,
        )
        activations = self._item_route_activations()
        total = jnp.sum(activations)
        return jnp.hstack(
            (
                p_stop,
                (1 - p_stop) * activations / lax.select(total == 0, 1.0, total),
            )
        )

    def _position_route_outcome_probabilities(
        self,
    ) -> Float[Array, " recall_outcomes"]:
        p_stop = self._route_stop_probability(
            self.position_recallable,
            self.position_mcf,
            self.studied > 0,
        )
        activations = self._position_route_item_activations()
        total = jnp.sum(activations)
        return jnp.hstack(
            (
                p_stop,
                (1 - p_stop) * activations / lax.select(total == 0, 1.0, total),
            )
        )

    def _position_route_reinstatement(self, item_index: Int_) -> Float[Array, " n"]:
        item_support = self._position_route_position_activations() * self._item_mask(
            item_index
        )
        normalized = self._normalize(item_support)
        cue = power_scale(normalized, self.mfc_sensitivity)
        return self.position_mfc.probe(cue)

    def experience_item(self, item_index: Int_) -> "CMR":
        """Return the model after studying the specified item."""
        item_cue = self.items[item_index]
        position_cue = self.positions[self.study_index]

        item_input = self.item_mfc.probe(item_cue)
        position_input = self.position_mfc.probe(position_cue)
        mixed_input = (
            self.blend_weight * item_input
            + (1 - self.blend_weight) * position_input
        )

        new_context = self.context.integrate(mixed_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )

        return self.replace(
            context=new_context,
            item_mfc=self.item_mfc.associate(
                item_cue, learning_state, self.mfc_learning_rate
            ),
            item_mcf=self.item_mcf.associate(
                learning_state, item_cue, self.mcf_learning_rate
            ),
            position_mfc=self.position_mfc.associate(
                position_cue, learning_state, self.mfc_learning_rate
            ),
            position_mcf=self.position_mcf.associate(
                learning_state, position_cue, self.mcf_learning_rate
            ),
            item_studied=self.item_studied.at[item_index].set(True),
            item_recallable=self.item_recallable.at[item_index].set(True),
            studied=self.studied.at[self.study_index].set(item_index + 1),
            position_recallable=self.position_recallable.at[self.study_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> "CMR":
        """Return model after simulating the specified study event."""
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "CMR":
        """Return model after transitioning from study to retrieval mode."""
        start_context = self.context.integrate(
            self.context.initial_state, self.start_drift_rate
        )
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of the specified item."""
        item_outcomes = self._item_route_outcome_probabilities()
        position_outcomes = self._position_route_outcome_probabilities()

        item_choice_prob = item_outcomes[item_index + 1]
        position_choice_prob = position_outcomes[item_index + 1]
        mixed_choice_prob = (
            self.blend_weight * item_choice_prob
            + (1 - self.blend_weight) * position_choice_prob
        )

        posterior_item_weight = lax.cond(
            mixed_choice_prob > 0,
            lambda: (self.blend_weight * item_choice_prob) / mixed_choice_prob,
            lambda: self.blend_weight,
        )

        item_reinstatement = self.item_mfc.probe(self.items[item_index])
        position_reinstatement = self._position_route_reinstatement(item_index)
        reinstatement = (
            posterior_item_weight * item_reinstatement
            + (1 - posterior_item_weight) * position_reinstatement
        )

        new_context = self.context.integrate(reinstatement, self.recall_drift_rate)

        new_item_recallable = self.item_recallable.at[item_index].set(
            self.allow_repeated_recalls
        )
        new_position_recallable = lax.cond(
            self.allow_repeated_recalls,
            lambda: self.position_recallable,
            lambda: self.position_recallable * (self.studied != item_index + 1),
        )

        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            item_recallable=new_item_recallable,
            position_recallable=new_position_recallable,
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "CMR":
        """Return model after simulating the specified retrieval event."""
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def position_activations(self) -> Float[Array, " list_length"]:
        """Return positional-route support over study positions."""
        return self._position_route_position_activations()

    def activations(self) -> Float[Array, " item_count"]:
        """Return blended item support from the two route laws."""
        return (
            self.blend_weight * self._item_route_activations()
            + (1 - self.blend_weight) * self._position_route_item_activations()
        )

    def stop_probability(self) -> Float[Array, ""]:
        """Return the mixed termination probability."""
        return self.outcome_probabilities()[0]

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return mixed conditional probability of recalling the specified item."""
        outcomes = self.outcome_probabilities()
        p_continue = 1 - outcomes[0]
        return lax.cond(
            p_continue == 0,
            lambda: 0.0,
            lambda: outcomes[item_index + 1] / p_continue,
        )

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return probability of the specified recall outcome."""
        return self.outcome_probabilities()[choice]

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return the mixed observable outcome distribution."""
        item_outcomes = self._item_route_outcome_probabilities()
        position_outcomes = self._position_route_outcome_probabilities()
        return (
            self.blend_weight * item_outcomes
            + (1 - self.blend_weight) * position_outcomes
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
