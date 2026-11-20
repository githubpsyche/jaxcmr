"""Context Maintenance and Retrieval (CMR) represents each item once per trial.
Studying an item moves temporal context toward that item's associated context
and strengthens the associations in both directions. A second presentation
updates the same item's associations.

At recall, the current context cues the available items. By default CMR
chooses whether to stop, then selects an item in proportion to its support.
Each policy returns the probabilities of STOP and the individual items.
Simulation samples from them and likelihood reads the observed entry. Recalling
an item reinstates its associated context, changing the cue for the next response.
"""

from collections.abc import Callable
from functools import partial
from typing import Self

import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.response import PositionalStop, relative_choice_weights
from jaxcmr.math import exponential_primacy_decay
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
    """Maintain item associations and temporal context for one recall trial."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        *,
        response_policy_create_fn: ResponsePolicyCreateFn["CMR"] = PositionalStop,
    ) -> None:
        """Context Maintenance and Retrieval model of memory search.

        Parameters
        ----------
        list_length : int
            Capacity for within-list item codes 1 through ``list_length``.
        parameters : Parameters
            Values governing study, retrieval and the selected components.
            See the parameter names below.
        mfc_create_fn : MemoryCreateFn, optional
            Factory function for item-to-context memory (M_FC).
        mcf_create_fn : MemoryCreateFn, optional
            Factory function for context-to-item memory (M_CF).
        context_create_fn : ContextCreateFn, optional
            Factory function for temporal context representation.
        response_policy_create_fn : ResponsePolicyCreateFn[CMR], optional
            Score STOP and every item. The default is ``PositionalStop``.

        Notes
        -----
        - Context drift. ``encoding_drift_rate``, ``start_drift_rate``, and
          ``recall_drift_rate`` govern context change during study, at the start
          of retrieval, and after each recalled item, respectively.
        - Learning. ``learn_after_context_update`` selects the context after
          the study update when true and before it when false. ``learning_rate``
          sets item-to-context learning. ``primacy_scale`` and ``primacy_decay``
          set context-to-item learning by study position.
        - Recall. ``allow_repeated_recalls`` keeps a recalled item available.
          The default response rule uses ``choice_sensitivity`` to weight the
          eligible items. Other rules read their own response parameters.
        - Default components. Linear memory uses ``shared_support`` and
          ``item_support`` for initial context-to-item associations. Positional
          stopping uses ``stop_probability_scale`` and ``stop_probability_growth``.

        """
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        self.items = jnp.eye(list_length)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.response_policy = response_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.studied = jnp.zeros(self.item_count, dtype=bool)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """Learning rate for context-to-item memory at current study position.

        Returns
        -------
        Float[Array, ""]
            Position-dependent learning rate from primacy gradient.

        """
        return self._mcf_learning_rate[self.study_index]

    def experience_item(self, item_index: Int_) -> Self:
        """Simulate encoding of an item during study, updating context and memories.

        Parameters
        ----------
        item_index : Int_
            Index of the item to experience (0-indexed).

        Returns
        -------
        CMR
            Updated model state after encoding the item.

        """
        item = self.items[item_index]
        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
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
        """Simulate a study event.

        Parameters
        ----------
        choice : Int_
            Index of item to experience (1-indexed). A value of 0 is
            ignored and returns the model unchanged.

        Returns
        -------
        CMR
            Updated model state after the study event.

        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> Self:
        """Transition from study to retrieval mode by reinstating start context.

        Returns
        -------
        CMR
            Model state ready to begin retrieval.

        """
        start_input = self.context.initial_state
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> Self:
        """Simulate retrieval of a specific item, updating context and records.

        Parameters
        ----------
        item_index : Int_
            Index of the item to retrieve (0-indexed).

        Returns
        -------
        CMR
            Updated model state after retrieval.

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

    def retrieve(self, choice: Int_) -> Self:
        """Simulate a retrieval event.

        Parameters
        ----------
        choice : Int_
            Index of item to retrieve (1-indexed), or 0 to terminate
            recall.

        Returns
        -------
        CMR
            Updated model state after the retrieval event.

        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def item_evidence(self) -> Float[Array, " items"]:
        """Return the memory probe, in item-code order, before any choice transform."""
        return self.mcf.probe(self.context.state)

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Weight the selected studied items by their relative memory evidence."""
        return relative_choice_weights(
            self.item_evidence(), candidates & self.studied, sensitivity
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return the policy's distribution in the order STOP, item 1, item 2, …."""
        return self.response_policy.outcome_probabilities(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Score the observed outcome. Zero denotes STOP, positive codes denote items."""
        return self.outcome_probabilities()[choice]


def make_factory(
    mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
    mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
    context_create_fn: ContextCreateFn = TemporalContext.init,
    *,
    response_policy_create_fn: ResponsePolicyCreateFn[CMR] = PositionalStop,
) -> type[MemorySearchModelFactory]:
    """Create a CMR model factory with specified component functions.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Factory function for item-to-context memory (M_FC).
    mcf_create_fn : MemoryCreateFn
        Factory function for context-to-item memory (M_CF).
    context_create_fn : ContextCreateFn
        Factory function for temporal context representation.
    response_policy_create_fn : ResponsePolicyCreateFn[CMR], optional
        Build the complete response rule. For example, pass
        ``PowerLawStopCompetition`` or ``RawSupportRatioSoftmaxChoice``.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class for a loss or simulator. Instantiate it with a dataset
        and optional features. Each call to ``create_trial_model`` creates a
        fresh model with the dataset's maximum list length. Ordinary CMR uses
        the same initialization for every trial and ignores vocabulary features.

    """

    return _trial_factory(
        partial(
            CMR,
            mfc_create_fn=mfc_create_fn,
            mcf_create_fn=mcf_create_fn,
            context_create_fn=context_create_fn,
            response_policy_create_fn=response_policy_create_fn,
        )
    )


def _trial_factory(
    create: Callable[[int, Parameters], MemorySearch],
) -> type[MemorySearchModelFactory]:
    """Use one item capacity for every trial in a dataset."""

    class CMRModelFactory:
        def __init__(
            self,
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            self.max_list_length = np.max(dataset["listLength"]).item()
            self.model_create_fn = create

        def create_model(self, parameters: Parameters) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self, trial_index: Integer[Array, ""], parameters: Parameters
        ) -> MemorySearch:
            return self.create_model(parameters)

    return CMRModelFactory
