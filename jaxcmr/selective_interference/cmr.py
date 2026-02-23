"""CMR variant for selective interference simulations.

Defines ``PhasedCMR``, a standalone Pytree model with per-phase
encoding methods for selective interference paradigms.  Each
experimental phase (film, break, interference, filler) has a
dedicated ``experience_*`` method that uses phase-specific drift
and MCF learning rates stored at construction.

The shared ``experience`` method accepts drift rate and MCF scale
as arguments, providing the common encoding logic that all
phase-specific methods delegate to.

"""

from typing import Callable, Mapping, Optional

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
    MemoryCreateFn,
    TerminationPolicyCreateFn,
)


class PhasedCMR(Pytree):
    """CMR with per-phase encoding for selective interference."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        is_emotional: Optional[Float[Array, " items"]] = None,
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    ):
        """Initialize phased CMR for selective interference.

        Parameters
        ----------
        list_length : int
            Number of items in the study list.
        parameters : Mapping[str, Float_]
            Model parameters.  Phase-specific keys
            (``break_drift_rate``, ``break_mcf_scale``, etc.)
            default to base-model values when absent.
        is_emotional : Float[Array, " items"] or None, optional
            Per-item emotional flag (1.0 = emotional, 0.0 = neutral).
            Defaults to all-zeros (no emotional items).
        mfc_create_fn : MemoryCreateFn, optional
            Factory function for item-to-context memory.
        mcf_create_fn : MemoryCreateFn, optional
            Factory function for context-to-item memory.
        context_create_fn : ContextCreateFn, optional
            Factory function for temporal context.
        termination_policy_create_fn : TerminationPolicyCreateFn, optional
            Factory function for recall termination policy.

        """
        # Core model params
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]

        # Emotional mechanism (defaults off when is_emotional is None or emotion_scale is 0)
        self.emotion_scale = parameters.get("emotion_scale", 0.0)
        _is_emo = is_emotional if is_emotional is not None else jnp.zeros(list_length)
        self.is_emotional = jnp.array(_is_emo, dtype=jnp.float32)
        self.emotional_mcf = jnp.array(_is_emo, dtype=jnp.float32)

        # Phase-specific params (default to base rates when absent)
        self.break_drift_rate = parameters.get(
            "break_drift_rate", self.encoding_drift_rate
        )
        self.break_mcf_scale = parameters.get("break_mcf_scale", 1.0)
        self.interference_drift_rate = parameters.get(
            "interference_drift_rate", self.encoding_drift_rate
        )
        self.interference_mcf_scale = parameters.get("interference_mcf_scale", 1.0)
        self.filler_drift_rate = parameters.get(
            "filler_drift_rate", self.encoding_drift_rate
        )
        self.filler_mcf_scale = parameters.get("filler_mcf_scale", 1.0)
        self.reminder_start_drift_rate = parameters.get(
            "reminder_start_drift_rate", self.start_drift_rate
        )
        self.reminder_drift_rate = parameters.get(
            "reminder_drift_rate", self.encoding_drift_rate
        )

        # Model state
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.termination_policy = termination_policy_create_fn(
            list_length, parameters
        )
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """Learning rate for context-to-item memory at current position.

        Returns
        -------
        Float[Array, ""]

        """
        return exponential_primacy_decay(
            self.study_index, self.primacy_scale, self.primacy_decay
        )

    # ── Shared encoding engine 
    
    def experience_item(
        self,
        item_index: Int_,
        drift_rate: Float_,
        mcf_lr_scale: Float_,
    ) -> "PhasedCMR":
        """Simulate encoding of an item during study, updating context and memories.

        Parameters
        ----------
        item_index : Int_
            Index of the item to experience (0-indexed).
        drift_rate : Float_
            Context drift rate for this encoding step.
        mcf_lr_scale : Float_
            Multiplier on the MCF learning rate.

        Returns
        -------
        CMR
            Updated model state after encoding the item.

        """
        item = self.items[item_index]
        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )
        # Emotional MCF encoding (eeg_ecmr pattern minus LPP/primacy)
        emcf_lr = self.emotion_scale * self.is_emotional[item_index]
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(item, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(
                learning_state, item, mcf_lr_scale * self.mcf_learning_rate
            ),
            emotional_mcf=self.emotional_mcf.at[item_index].multiply(emcf_lr),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(
        self,
        choice: Int_,
        drift_rate: Float_,
        mcf_lr_scale: Float_,
    ) -> "PhasedCMR":
        """Encode a study item with specified rates.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.
        drift_rate : Float_
            Context drift rate for this encoding step.
        mcf_lr_scale : Float_
            Multiplier on the MCF learning rate.

        Returns
        -------
        PhasedCMR

        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1, drift_rate, mcf_lr_scale),
        )

    # ── Phase-specific encoding ─────────────────────────────────────

    def experience_film(self, choice: Int_) -> "PhasedCMR":
        """Encode a film-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedCMR

        """
        return self.experience(choice, self.encoding_drift_rate, 1.0)

    def experience_break(self, choice: Int_) -> "PhasedCMR":
        """Encode a break-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedCMR

        """
        return self.experience(choice, self.break_drift_rate, self.break_mcf_scale)

    def experience_interference(self, choice: Int_) -> "PhasedCMR":
        """Encode an interference-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedCMR

        """
        return self.experience(
            choice, self.interference_drift_rate, self.interference_mcf_scale
        )

    def experience_filler(self, choice: Int_) -> "PhasedCMR":
        """Encode a filler-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedCMR

        """
        return self.experience(
            choice, self.filler_drift_rate, self.filler_mcf_scale
        )

    # ── Context operations ──────────────────────────────────────────

    def start_reminders(self) -> "PhasedCMR":
        """Transition to reminder phase.

        Returns
        -------
        PhasedCMR

        """
        new_context = self.context.integrate(
            self.context.initial_state, self.reminder_start_drift_rate
        )
        return self.replace(context=new_context)

    def remind(self, choice: Int_) -> "PhasedCMR":
        """Replay a single item's context association without learning.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedCMR

        """

        def _remind_item(self_inner):
            item = self_inner.items[choice - 1]
            context_input = self_inner.mfc.probe(item)
            new_context = self_inner.context.integrate(
                context_input, self_inner.reminder_drift_rate
            )
            return self_inner.replace(context=new_context)

        return lax.cond(choice == 0, lambda: self, lambda: _remind_item(self))

    # ── Retrieval ───────────────────────────────────────────────────

    def start_retrieving(self) -> "PhasedCMR":
        """Transition from study to retrieval mode.

        Returns
        -------
        PhasedCMR

        """
        start_context = self.context.integrate(
            self.context.initial_state, self.start_drift_rate
        )
        return self.replace(context=start_context)

    def retrieve(self, choice: Int_) -> "PhasedCMR":
        """Simulate a retrieval event.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed), or 0 to terminate.

        Returns
        -------
        PhasedCMR

        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self._retrieve_item(choice - 1),
        )

    def _retrieve_item(self, item_index: Int_) -> "PhasedCMR":
        new_context = self.context.integrate(
            self.mfc.probe(self.items[item_index]),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(
                self.allow_repeated_recalls
            ),
            recall_total=self.recall_total + 1,
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations for all items.

        Returns
        -------
        Float[Array, " item_count"]

        """
        _activations = self.mcf.probe(self.context.state) * self.recallable

        # Emotional clustering: boost emotional items when last recall was emotional
        last_emotional = lax.cond(
            self.recall_total > 0,
            lambda: self.is_emotional[self.recalls[self.recall_total - 1] - 1] > 0,
            lambda: False,
        )
        emotion_support = last_emotional * self.emotional_mcf

        return (
            power_scale(_activations + emotion_support, self.mcf_sensitivity) + lb
        ) * self.recallable

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
            Index of item (1-indexed), or 0 for termination.

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
                    / lax.select(
                        item_activation_sum == 0, 1.0, item_activation_sum
                    )
                ),
            )
        )


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    termination_policy_create_fn: TerminationPolicyCreateFn = PositionalTermination,
    is_emotional: Optional[Float[Array, " items"]] = None,
) -> Callable:
    """Build a PhasedCMR factory compatible with prepare_sweep.

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
    is_emotional : Float[Array, " items"] or None, optional
        Per-item emotional flag (1.0 = emotional, 0.0 = neutral).
        Captured in the closure; defaults to all-zeros.

    Returns
    -------
    Callable

    """
    def factory(
        list_length: int,
        parameters: Mapping[str, Float_],
        connections: Optional[Float[Array, " n n"]] = None,
    ) -> PhasedCMR:
        if is_emotional is None:
            _is_emotional = jnp.zeros(list_length)
        else:
            # Pad to list_length for extended tiers (extra slots are neutral).
            _is_emotional = jnp.zeros(list_length).at[
                :is_emotional.shape[0]
            ].set(is_emotional)
        return PhasedCMR(
            list_length,
            parameters,
            is_emotional=_is_emotional,
            mfc_create_fn=mfc_create_fn,
            mcf_create_fn=mcf_create_fn,
            context_create_fn=context_create_fn,
            termination_policy_create_fn=termination_policy_create_fn,
        )
    return factory
