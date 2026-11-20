"""CMR with distinct positional contexts.

Items are encoded using positional representations rather than item
representations. This allows repeated items to have distinct
contextual associations for each presentation. Recallability is
tracked at the item level (not position level).

"""



import numpy as np
from jax import lax
from jax import numpy as jnp

import jaxcmr.components.context as TemporalContext
import jaxcmr.components.linear_memory as LinearMemory
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.response import PositionalStop, pool_position_weights
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
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
        mfc_create_fn: MemoryCreateFn = LinearMemory.init_mfc,
        mcf_create_fn: MemoryCreateFn = LinearMemory.init_mcf,
        context_create_fn: ContextCreateFn = TemporalContext.init,
        response_policy_create_fn: ResponsePolicyCreateFn["CMR"] = PositionalStop,
    ) -> None:
        self.encoding_drift_rate = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.learn_after_context_update = parameters["learn_after_context_update"]
        self.allow_repeated_recalls = parameters["allow_repeated_recalls"]
        self.item_count = list_length
        #! item representations on F now position representations
        self.positions = jnp.eye(self.item_count)
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        #! We track studied item for each study position
        self.item_ids = jnp.arange(list_length)
        self.studied = jnp.zeros(list_length, dtype=int)
        self.context = context_create_fn(list_length)
        self.mfc = mfc_create_fn(list_length, parameters, self.context)
        self.mcf = mcf_create_fn(list_length, parameters, self.context)
        self.response_policy = response_policy_create_fn(list_length, parameters)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """The learning rate for the MCF memory under its current state."""
        return self._mcf_learning_rate[self.study_index]

    def experience_item(self, item_index: Int_) -> "CMR":
        """Return the model after experiencing item with the specified index.

        Args:
            item_index: the index of the item to experience. 0-indexed.
        """
        #! instead of probing and learning using item, we use the item's study position
        mfc_cue = self.positions[self.study_index]
        context_input = self.mfc.probe(mfc_cue)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)
        learning_state = lax.cond(
            self.learn_after_context_update,
            lambda: new_context.state,
            lambda: self.context.state,
        )
        return self.replace(
            context=new_context,
            #! associate using position cue instead of item
            mfc=self.mfc.associate(mfc_cue, learning_state, self.mfc_learning_rate),
            mcf=self.mcf.associate(learning_state, mfc_cue, self.mcf_learning_rate),
            recallable=self.recallable.at[item_index].set(True),
            #! track each item's study position(s)
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
        start_input = self.context.initial_state
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of item with the specified index.

        Args:
            choice: the index of the item to retrieve (0-indexed)
        """
        #! Pool item activation across the item's study positions
        new_context = self.context.integrate(
            self.mfc.probe(self.studied == item_index + 1),
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

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Pool positions by item, apply relative power across items, then select candidates."""
        evidence = pool_position_weights(
            self.mcf.probe(self.context.state), self.studied
        )
        present = jnp.any(self.studied[:, None] == self.item_ids[None, :] + 1, axis=0)
        return (relative_power_scale(evidence, sensitivity) + FLOAT32_EPS) * (
            candidates & present
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return probabilities in recorded order: STOP, then each local item."""
        return self.response_policy.outcome_probabilities(self)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Score the recorded STOP or item using the same distribution as simulation."""
        return self.outcome_probabilities()[choice]


def make_factory(
    mfc_create_fn: MemoryCreateFn,
    mcf_create_fn: MemoryCreateFn,
    context_create_fn: ContextCreateFn,
    response_policy_create_fn: ResponsePolicyCreateFn[CMR],
) -> type[MemorySearchModelFactory]:
    """Encode distinct study positions and track recall availability by item.

    Parameters
    ----------
    mfc_create_fn : MemoryCreateFn
        Construct the memory probed to reinstate context.
    mcf_create_fn : MemoryCreateFn
        Construct the memory probed for candidate support.
    context_create_fn : ContextCreateFn
        Construct the temporal context for a trial.
    response_policy_create_fn : ResponsePolicyCreateFn[CMR]
        The policy assigns probability to STOP and each recorded item.

    Returns
    -------
    type[MemorySearchModelFactory]
        A factory class initialized with the dataset and any semantic inputs.
    """

    class CMRModelFactory:
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
                    response_policy_create_fn,
                )

            self.model_create_fn = model_create_fn

        def create_model(self, parameters: Parameters) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

        def create_trial_model(
            self,
            trial_index: Integer[Array, ""],
            parameters: Parameters,
        ) -> MemorySearch:
            return self.model_create_fn(self.max_list_length, parameters)

    return CMRModelFactory
