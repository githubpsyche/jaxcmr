"""A terminal-item CMR trial learns an association for the end of the list.

For a study list [A, B, C], the associative memories represent [A, B, C, END].
At the end of study the model associates END with the current context. During
retrieval the default response policy uses the evidence for END to score STOP.
The recorded outcomes are STOP (0) and the three study-item codes (1–3).
"""

from functools import partial
from typing import Self

from jax import lax
from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components import context, linear_memory
from jaxcmr.components.response import LearnedStopCompetition, relative_choice_weights
from jaxcmr.math import exponential_primacy_decay
from jaxcmr.models.cmr import _trial_factory
from jaxcmr.typing import (
    Array,
    Bool,
    ContextCreateFn,
    Float,
    Float_,
    Int_,
    MemoryCreateFn,
    MemorySearchModelFactory,
    Parameters,
    ResponsePolicyCreateFn,
)

__all__ = ["TerminalItemCMR", "make_factory"]


class TerminalItemCMR(Pytree):
    """CMR with a private memory slot for the end of the list.

    The retrieval-start update associates that slot with the final study
    context before reinstating the start cue. Both terminal associations use
    ``learning_rate``. By default its weight competes on the same relative-power
    scale as eligible items. Only studied items receive recall codes.
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        mfc_create_fn: MemoryCreateFn = linear_memory.init_mfc,
        mcf_create_fn: MemoryCreateFn = linear_memory.init_mcf,
        context_create_fn: ContextCreateFn = context.init,
        *,
        response_policy_create_fn: ResponsePolicyCreateFn[
            "TerminalItemCMR"
        ] = LearnedStopCompetition,
    ) -> None:
        self.item_count = list_length
        # The last item represents END. Only the study items have recall codes.
        memory_item_count = list_length + 1
        self.items = jnp.eye(memory_item_count)
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(memory_item_count, parameters, self.context)
        self.mcf = mcf_create_fn(memory_item_count, parameters, self.context)
        self.response_policy = response_policy_create_fn(list_length, parameters)

        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.recalls = jnp.zeros(list_length, dtype=int)
        self.studied = jnp.zeros(list_length, dtype=bool)
        self.recallable = jnp.zeros(list_length, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """Return the primacy weight for the next study position."""
        return self._mcf_learning_rate[self.study_index]

    def experience_item(self, item_index: Int_) -> Self:
        """Encode the zero-indexed study item and associate it with context."""
        item = self.items[item_index]
        new_context = self.context.integrate(self.mfc.probe(item), self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update, lambda: new_context.state, lambda: self.context.state
        )
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(item, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(learning_state, item, self.mcf_learning_rate),
            studied=self.studied.at[item_index].set(True),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
        )

    def experience(self, choice: Int_) -> Self:
        """Study a one-indexed item code. Zero denotes padding."""
        return lax.cond(choice == 0, lambda: self, lambda: self.experience_item(choice - 1))

    def start_retrieving(self) -> Self:
        """Associate END with the final study context, then reinstate the start cue."""
        terminal = self.items[-1]
        final_context = self.context.state
        return self.replace(
            mfc=self.mfc.associate(terminal, final_context, self.mfc_learning_rate),
            mcf=self.mcf.associate(final_context, terminal, self.mfc_learning_rate),
            context=self.context.integrate(self.context.initial_state, self.start_drift_rate),
        )

    def retrieve_item(self, item_index: Int_) -> Self:
        """Recall the zero-indexed item and reinstate its associated context."""
        context_input = self.mfc.probe(self.items[item_index])
        return self.replace(
            context=self.context.integrate(context_input, self.recall_drift_rate),
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(self.allow_repeated_recalls),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> Self:
        """Record an item recall, or end retrieval on STOP (code zero)."""
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def item_evidence(self) -> Float[Array, " items"]:
        """Return memory evidence for the study items in item-code order."""
        return self.mcf.probe(self.context.state)[:-1]

    def terminal_and_item_evidence(self) -> Float[Array, " outcomes"]:
        """Return the raw memory probe in outcome order, with STOP first."""
        evidence = self.mcf.probe(self.context.state)
        return jnp.append(evidence[-1], evidence[:-1])

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Weight the selected studied items by their relative memory evidence."""
        return relative_choice_weights(
            self.item_evidence(), candidates & self.studied, sensitivity
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Score STOP followed by each study item with the selected response policy."""
        return self.response_policy.outcome_probabilities(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return the probability of the observed STOP or item code."""
        return self.outcome_probabilities()[choice]


def make_factory(
    mfc_create_fn: MemoryCreateFn = linear_memory.init_mfc,
    mcf_create_fn: MemoryCreateFn = linear_memory.init_mcf,
    context_create_fn: ContextCreateFn = context.init,
    *,
    response_policy_create_fn: ResponsePolicyCreateFn[
        TerminalItemCMR
    ] = LearnedStopCompetition,
) -> type[MemorySearchModelFactory]:
    """Encode a terminal item at list end and use the supplied response rule.

    Memory constructors receive one extra item slot. Context and recorded
    outcomes retain their ordinary dimensions. The default uses proportional
    competition between the learned terminal candidate and studied items.
    """
    return _trial_factory(
        partial(
            TerminalItemCMR,
            mfc_create_fn=mfc_create_fn,
            mcf_create_fn=mcf_create_fn,
            context_create_fn=context_create_fn,
            response_policy_create_fn=response_policy_create_fn,
        )
    )
