"""CMR3 with valence-coded source context.

Implements the Cohen and Kahana (2022) CMR3-style emotional
context model with dual context layers (temporal and emotional),
2-unit valence source features, and per-item encoding-strength
modulation via emotion category.

Notes
-----
The temporal pathway uses standard CMR encoding and retrieval.
The emotional pathway maintains a separate 3-D emotional context
(start-of-list + negative pole + positive pole) and separate
emotion_mfc / emotion_mcf association matrices. During encoding,
the emotional context-to-item learning rate is scaled by phi_emot
(which incorporates whether an item is non-neutral), while the
temporal pathway learning rate depends only on primacy unless
phi_emot_modulates_temporal is True.

Negative and positive items have distinct source features and
update emotional context. Neutral items have no direct valence
feature and therefore do not inject new valence input into the
emotional context pathway.
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
    "CMR3",
    "make_factory",
]


class CMR3(Pytree):
    """Full CMR3 with dual context and valence-coded source channels."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_negative: Bool[Array, " study_events"],
        is_positive: Bool[Array, " study_events"],
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ) -> None:
        """Initialize full CMR3 with dual context and valence coding."""
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
        self.is_negative = jnp.array(is_negative, dtype=jnp.float32)
        self.is_positive = jnp.array(is_positive, dtype=jnp.float32)
        self.is_emotional = jnp.maximum(self.is_negative, self.is_positive)

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
        # 3-D emotional context: [start-of-list, negative_pole, positive_pole]
        self.emotion_context = TemporalContext.TemporalContext(item_count=2, size=3)
        # emotion_mfc: item -> emotional context (list_length x 3)
        emot_mfc_state = jnp.zeros((list_length, 3))
        emot_mfc_state = emot_mfc_state.at[:, 1].set(
            (1 - self.mfc_learning_rate) * self.is_negative
        )
        emot_mfc_state = emot_mfc_state.at[:, 2].set(
            (1 - self.mfc_learning_rate) * self.is_positive
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

    def experience_item(self, item_index: Int_) -> "CMR3":
        """Encode a single item, updating both context pathways."""
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

    def experience(self, choice: Int_) -> "CMR3":
        """Encode a study item."""
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "CMR3":
        """Transition from study to retrieval mode."""
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

    def _retrieve_item(self, item_index: Int_) -> "CMR3":
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

    def retrieve(self, choice: Int_) -> "CMR3":
        """Simulate a retrieval event."""
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self._retrieve_item(choice - 1),
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations combining both pathways."""
        temporal_act = self.mcf.probe(self.context.state) * self.recallable
        emotional_act = self.emotion_mcf.probe(self.emotion_context.state) * self.recallable
        combined = temporal_act + emotional_act
        return (power_scale(combined, self.mcf_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Compute probability of terminating recall."""
        return self.termination_policy.stop_probability(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Compute probability of a specific retrieval outcome."""
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
        """Compute probabilities for all retrieval outcomes."""
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
    """Build a CMR3 factory for the CK2022 fitting pipeline."""

    class CMR3ModelFactory:
        """Factory creating trial-specific CMR3 instances from CK2022 data."""

        def __init__(
            self,
            dataset: RecallDataset,
            features: Optional[Float[Array, " word_pool_items features_count"]],
        ) -> None:
            self.present_lists = jnp.array(dataset["pres_itemids"])
            self.max_list_length = np.max(dataset["listLength"]).item()

            valence = jnp.array(dataset["valence"])
            self.trial_negative = jnp.array(valence < 0, dtype=bool)
            self.trial_positive = jnp.array(valence > 0, dtype=bool)

            def model_create_fn(
                list_length: int,
                parameters: Mapping[str, Float_],
                is_negative: Bool[Array, " study_events"],
                is_positive: Bool[Array, " study_events"],
            ) -> MemorySearch:
                return CMR3(
                    list_length,
                    parameters,
                    is_negative,
                    is_positive,
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
                self.trial_negative[0],
                self.trial_positive[0],
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
                self.trial_negative[trial_index],
                self.trial_positive[trial_index],
            )

    return CMR3ModelFactory
