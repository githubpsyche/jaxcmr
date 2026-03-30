"""eCMR with category-based emotional encoding strength.

Implements the full emotional context maintenance and retrieval
model (eCMR; Talmi, Lohnas, & Daw, 2019) with dual context layers
(temporal and emotional), 2-unit localist source features, and
per-item encoding-strength modulation via emotion category.

Notes
-----
The temporal pathway uses standard CMR encoding and retrieval.
The emotional pathway maintains a separate 3-D emotional context
(start-of-list + emotional pole + neutral pole) and separate
emotion_mfc / emotion_mcf association matrices. During encoding,
the emotional context-to-item learning rate is scaled by phi_emot
(which incorporates emotion category), while the temporal pathway
learning rate depends only on primacy unless
phi_emot_modulates_temporal is True.

Both emotional and neutral items have source features and update
emotional context. Emotional items push emotional context toward
the emotional pole; neutral items push it toward the neutral pole.
This 2-unit localist representation is critical for the
list-composition effect.
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
    Bool,
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
    "eCMR",
    "make_factory",
]


class eCMR(Pytree):
    """Full eCMR with dual context and category-modulated encoding."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_emotional: Bool[Array, " study_events"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ) -> None:
        """Initialize full eCMR with dual context and category modulation.

        Parameters
        ----------
        list_length : int
            Number of items in the study list.
        parameters : Mapping[str, Float_]
            Model parameters.
        is_emotional : Bool[Array, " study_events"]
            Per-item emotional flag (1 = emotional, 0 = neutral).
        mfc_create_fn : MemoryCreateFn, optional
            Factory for item-to-context memory.
        mcf_create_fn : MemoryCreateFn, optional
            Factory for context-to-item memory.
        context_create_fn : ContextCreateFn, optional
            Factory for temporal context.
        termination_policy_create_fn : TerminationPolicyCreateFn, optional
            Factory for recall termination policy.

        """
        # --- Parameters ---
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.emotion_drift_rate = parameters.get("emotion_drift_rate", 1.0)
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.modulate_emotion_by_primacy = parameters["modulate_emotion_by_primacy"]
        self.phi_emot_modulates_temporal = parameters.get(
            "phi_emot_modulates_temporal", False
        )
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]

        # --- Item representations ---
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)
        self.is_emotional = jnp.array(is_emotional, dtype=jnp.float32)

        # --- Primacy ---
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.primacy = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )

        # --- Phi_emot ---
        # Scales L^CF_sw (emotional pathway); also temporal when
        # phi_emot_modulates_temporal is True.
        emotion_scale = parameters["emotion_scale"]
        self.phi_emot = emotion_scale * self.is_emotional

        # --- Temporal pathway ---
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)

        # --- Emotional pathway ---
        #! Hardcoded inits — not using create_fn factories yet
        # 3-D emotional context: [start-of-list, emotional_pole, neutral_pole]
        self.emotion_context = TemporalContext.TemporalContext(item_count=2, size=3)
        # emotion_mfc: item -> emotional context (list_length x 3)
        is_neutral = 1.0 - self.is_emotional
        emot_mfc_state = jnp.zeros((list_length, 3))
        emot_mfc_state = emot_mfc_state.at[:, 1].set(
            (1 - self.mfc_learning_rate) * self.is_emotional
        )
        emot_mfc_state = emot_mfc_state.at[:, 2].set(
            (1 - self.mfc_learning_rate) * is_neutral
        )
        self.emotion_mfc = LinearMemory.LinearMemory(emot_mfc_state)
        # emotion_mcf: emotional context -> item (3 x list_length, all zeros)
        self.emotion_mcf = LinearMemory.LinearMemory(jnp.zeros((3, list_length)))

        # --- Recall state ---
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.studied = jnp.zeros(self.item_count, dtype=bool)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    def experience_item(self, item_index: Int_) -> "eCMR":
        """Encode a single item, updating both context pathways.

        Parameters
        ----------
        item_index : Int_
            Index of the item (0-indexed).

        Returns
        -------
        eCMR

        """
        item = self.items[item_index]

        # --- Temporal pathway ---
        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )

        # --- Emotional pathway ---
        # Both emotional and neutral items update emotional context
        emot_context_input = self.emotion_mfc.probe(item)
        new_emotion_context = self.emotion_context.integrate(
            emot_context_input, self.emotion_drift_rate
        )
        emot_learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_emotion_context.state,
            lambda: self.emotion_context.state,
        )

        phi_emot_i = jnp.maximum(0.0, self.phi_emot[self.study_index])
        p = self.primacy[self.study_index]

        def _multiplicative():
            return p * phi_emot_i

        def _additive():
            return p + jnp.maximum(-p, phi_emot_i)

        # Temporal MCF: primacy only, or primacy + phi_emot when broad
        temporal_mcf_lr = lax.cond(
            self.phi_emot_modulates_temporal,
            _additive,
            lambda: p,
        )

        # Emotional MCF: primacy x phi_emot or primacy + phi_emot
        emotional_mcf_lr = lax.cond(
            self.modulate_emotion_by_primacy, _multiplicative, _additive
        )

        return self.replace(
            # Temporal pathway updates
            context=new_context,
            mfc=self.mfc.associate(item, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(learning_state, item, temporal_mcf_lr),
            # Emotional pathway updates
            emotion_context=new_emotion_context,
            emotion_mfc=self.emotion_mfc.associate(
                item, emot_learning_state, self.mfc_learning_rate
            ),
            emotion_mcf=self.emotion_mcf.associate(
                emot_learning_state, item, emotional_mcf_lr
            ),
            studied=self.studied.at[item_index].set(True),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> "eCMR":
        """Encode a study item.

        Parameters
        ----------
        choice : Int_
            Item index (1-indexed). 0 is ignored.

        Returns
        -------
        eCMR

        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "eCMR":
        """Transition from study to retrieval mode.

        Returns
        -------
        eCMR

        """
        start_context = self.context.integrate(
            self.context.initial_state, self.start_drift_rate
        )
        start_emotion_context = self.emotion_context.integrate(
            self.emotion_context.initial_state, self.start_drift_rate
        )
        return self.replace(
            context=start_context,
            emotion_context=start_emotion_context,
        )

    def _retrieve_item(self, item_index: Int_) -> "eCMR":
        item = self.items[item_index]
        # Temporal context reinstatement
        new_context = self.context.integrate(
            self.mfc.probe(item), self.recall_drift_rate
        )
        # Emotional context reinstatement
        new_emotion_context = self.emotion_context.integrate(
            self.emotion_mfc.probe(item), self.recall_drift_rate
        )
        return self.replace(
            context=new_context,
            emotion_context=new_emotion_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(self.allow_repeated_recalls),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "eCMR":
        """Simulate a retrieval event.

        Parameters
        ----------
        choice : Int_
            Item index (1-indexed), or 0 to terminate.

        Returns
        -------
        eCMR

        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self._retrieve_item(choice - 1),
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations combining both pathways.

        Returns
        -------
        Float[Array, " item_count"]

        """
        temporal_act = self.mcf.probe(self.context.state) * self.recallable
        emotional_act = self.emotion_mcf.probe(self.emotion_context.state) * self.recallable
        combined = temporal_act + emotional_act
        return (power_scale(combined, self.mcf_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Compute probability of terminating recall.

        Returns
        -------
        Float[Array, ""]

        """
        return self.termination_policy.stop_probability(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Compute probability of a specific retrieval outcome.

        Parameters
        ----------
        choice : Int_
            Item index (1-indexed), or 0 for termination.

        Returns
        -------
        Float[Array, ""]

        """
        p_stop = self.stop_probability()
        return lax.cond(
            choice == 0,
            lambda: p_stop,
            lambda: lax.cond(
                jnp.logical_or(p_stop == 1.0, ~self.recallable[choice - 1]),
                lambda: 0.0,
                lambda: (1 - p_stop) * self._item_probability(choice - 1),
            ),
        )

    def _item_probability(self, item_index: Int_) -> Float[Array, ""]:
        item_activations = self.activations()
        return item_activations[item_index] / jnp.sum(item_activations)

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Compute probabilities for all retrieval outcomes.

        Returns
        -------
        Float[Array, " recall_outcomes"]

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
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
) -> Type[MemorySearchModelFactory]:
    """Build an eCMR factory for the CK2022 fitting pipeline.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn, optional
        Factory for item-to-context memory.
    mcf_create_fn : MemoryCreateFn, optional
        Factory for context-to-item memory.
    context_create_fn : ContextCreateFn, optional
        Factory for temporal context.
    termination_policy_create_fn : TerminationPolicyCreateFn, optional
        Factory for recall termination policy.

    Returns
    -------
    Type[MemorySearchModelFactory]

    """

    class eCMRModelFactory:
        """Factory creating trial-specific eCMR instances from CK2022 data."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ) -> None:
            self.present_lists = jnp.array(dataset["pres_itemids"])
            self.max_list_length = np.max(dataset["listLength"]).item()

            # 0 for neutral study events, 1 for emotional study events
            self.trial_emotions = jnp.array(dataset["valence"] != 0, dtype=bool)

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_emotional: Bool[Array, " study_events"],
            ) -> MemorySearch:
                return eCMR(
                    list_length,
                    parameters,
                    is_emotional,
                    mfc_create_fn,
                    mcf_create_fn,
                    context_create_fn,
                    termination_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Mapping[str, Float_]) -> MemorySearch:
            """Create model from first trial (for shape inference)."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_emotions[0],
            )

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Mapping[str, Float_],
        ) -> MemorySearch:
            """Create model for a specific trial."""
            return self.model_create_fn(
                self.max_list_length,
                parameters,
                self.trial_emotions[trial_index],
            )

    return eCMRModelFactory
