"""Context Maintenance and Retrieval (CMR) represents each item once per trial.
Studying an item moves temporal context toward that item's associated context
and strengthens the associations in both directions. A second presentation
updates the same item's associations.

At recall, the current context cues the available items. The stopping policy
gives the probability of ending recall. Conditional on continuing, each item's
probability is proportional to its retrieval support. Recalling an item
reinstates its associated context, changing the cue for the next response.
"""



import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.termination import PositionalTermination
from jaxcmr.math import (
    FLOAT32_EPS,
    exponential_primacy_decay,
    relative_power_scale,
)
from jaxcmr.typing import (
    Array,
    Bool,
    ContextCreateFn,
    Float,
    Int_,
    Integer,
    MemoryCreateFn,
    MemorySearch,
    MemorySearchModelFactory,
    Parameters,
    RecallDataset,
    TerminationPolicyCreateFn,
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
        termination_policy_create_fn: TerminationPolicyCreateFn[
            "CMR"
        ] = PositionalTermination,
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
        termination_policy_create_fn : TerminationPolicyCreateFn[CMR], optional
            Build the stopping rule. The default is ``PositionalTermination``.

        Notes
        -----
        - Context drift. ``encoding_drift_rate``, ``start_drift_rate``, and
          ``recall_drift_rate`` govern context change during study, at the start
          of retrieval, and after each recalled item, respectively.
        - Learning. ``learn_after_context_update`` selects the context after
          the study update when true and before it when false. ``learning_rate``
          sets item-to-context learning. ``primacy_scale`` and ``primacy_decay``
          set context-to-item learning by study position.
        - Recall. ``choice_sensitivity`` is the exponent applied to retrieval
          support. ``allow_repeated_recalls`` keeps a recalled item available
          when true.
        - Default components. Linear memory uses ``shared_support`` and
          ``item_support`` for initial context-to-item associations. Positional
          stopping uses ``stop_probability_scale`` and ``stop_probability_growth``.

        """
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        # self.delay_drift_rate = parameters["delay_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        self.items = jnp.eye(self.item_count)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.termination_policy = termination_policy_create_fn(list_length, parameters)
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

    def experience_item(self, item_index: Int_) -> "CMR":
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

    def experience(self, choice: Int_) -> "CMR":
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

    def start_retrieving(self) -> "CMR":
        """Transition from study to retrieval mode by reinstating start context.

        Returns
        -------
        CMR
            Model state ready to begin retrieval.

        """
        # delay_input = jnp.mean(self.mcf.state, axis=1)
        start_input = self.context.initial_state
        # start_context = self.context.integrate(
        #     delay_input, self.delay_drift_rate
        # ).integrate(start_input, self.start_drift_rate)
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
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

    def retrieve(self, choice: Int_) -> "CMR":
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

    def candidate_activations(
        self, candidates: Bool[Array, " item_count"]
    ) -> Float[Array, " item_count"]:
        """Score a specified set of items against the current context.

        Parameters
        ----------
        candidates : Boolean array
            One entry per item. The largest support among the selected items
            sets the scale before applying ``choice_sensitivity``. A caller can
            include previously recalled items in this mask.

        Returns
        -------
        Float[Array, " item_count"]
            Relative retrieval support for each item.

        """
        _activations = self.mcf.probe(self.context.state) * candidates
        return (relative_power_scale(_activations, self.mcf_sensitivity) + FLOAT32_EPS) * candidates

    def activations(self) -> Float[Array, " item_count"]:
        """Compute retrieval activations for all items from context-to-item memory.

        Returns
        -------
        Float[Array, " item_count"]
            Relative retrieval support for each item.

        """
        return self.candidate_activations(self.recallable)

    def stop_probability(self) -> Float[Array, ""]:
        """Compute probability of terminating recall.

        Returns
        -------
        Float[Array, ""]
            Probability of stopping retrieval given current state.

        """
        return self.termination_policy.stop_probability(self)

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Compute probability of retrieving a specific item, assuming recall continues.

        Parameters
        ----------
        item_index : Int_
            Index of the item (0-indexed).

        Returns
        -------
        Float[Array, ""]
            Probability of retrieving the specified item.

        """
        item_activations = self.activations()
        return item_activations[item_index] / jnp.sum(item_activations)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Compute probability of a specific retrieval outcome.

        Parameters
        ----------
        choice : Int_
            Index of item to retrieve (1-indexed), or 0 for recall
            termination.

        Returns
        -------
        Float[Array, ""]
            Probability of the specified outcome.

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
        """Compute probabilities for all possible retrieval outcomes.

        Returns
        -------
        Float[Array, " recall_outcomes"]
            Probability distribution over outcomes, where index 0 is
            termination and indices 1 to item_count are item recalls.

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
    mfc_create_fn: MemoryCreateFn,
    mcf_create_fn: MemoryCreateFn,
    context_create_fn: ContextCreateFn,
    termination_policy_create_fn: TerminationPolicyCreateFn[CMR],
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
    termination_policy_create_fn : TerminationPolicyCreateFn[CMR]
        Factory function for recall termination policy.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class for a loss or simulator. Instantiate it with a dataset
        and optional features. Each call to ``create_trial_model`` creates a
        fresh model with the dataset's maximum list length. Ordinary CMR uses
        the same initialization for every trial and ignores vocabulary features.

    """

    class CMRModelFactory:
        """Factory for creating CMR model instances; sets `max_list_length` from dataset.

        Parameters
        ----------
        dataset : RecallDataset
            Dataset containing recall trials with list length information.
        features : Float[Array, " word_pool_items features_count"] or None
            Accepted for the shared factory interface and unused by ordinary CMR.

        """

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
                    termination_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Parameters) -> MemorySearch:
            """Create a CMR model instance with the given parameters.

            Parameters
            ----------
            parameters : Parameters
                Model parameters.

            Returns
            -------
            MemorySearch
                Initialized CMR model instance.

            """
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            """Create a CMR model instance for a specific trial.

            Parameters
            ----------
            trial_index : Integer[Array, ""]
                Index of the trial (unused in base CMR).
            parameters : Parameters
                Model parameters.

            Returns
            -------
            MemorySearch
                Initialized CMR model instance.

            """
            return self.model_create_fn(self.max_list_length, parameters)

    return CMRModelFactory
