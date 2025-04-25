from typing import Mapping, Optional

import numpy as np
from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from cru_to_cmr.context import TemporalContext
from cru_to_cmr.decision_probability import (
    compute_runner_probabilities,
    compute_runner_probability,
    race_diffusion_precompute,
)
from cru_to_cmr.helpers import lb
from cru_to_cmr.memory import LinearMemory
from cru_to_cmr.typing import (
    Array,
    Context,
    Float,
    Float_,
    Int_,
    Integer,
    Memory,
    MemorySearch,
    ProbabilityCalculator,
)


def power_scale(value: Float_, scale: Float_) -> Float:
    """Returns value scaled by the exponent factor using logsumexp trick."""
    log_activation = jnp.log(value)
    return lax.cond(
        jnp.logical_and(jnp.any(value != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: value,
        None,
    )


def exponential_primacy_decay(
    study_index: Int_, primacy_scale: Float_, primacy_decay: Float_
):
    """Returns the exponential primacy weighting for the specified study event.

    Args:
        study_index: the index of the study event.
        primacy_scale: the scale factor for primacy effect.
        primacy_decay: the decay factor for primacy effect.
    """
    return primacy_scale * jnp.exp(-primacy_decay * study_index) + 1


def exponential_stop_probability(
    stop_probability_scale: Float_, stop_probability_growth: Float_, recall_total: Int_
):
    """Returns the exponential stop probability for the specified recall event.

    Args:
        stop_probability_scale: the scale factor for stop probability.
        stop_probability_growth: the growth factor for stop probability.
        recall_total: the total number of items recalled.
    """
    return stop_probability_scale * jnp.exp(recall_total * stop_probability_growth)


class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        decision_strategy: ProbabilityCalculator,
    ):
        # base cmr parameters
        self.encoding_drift_rate_max = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.shared_support = parameters["shared_support"]
        self.item_support = parameters["item_support"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]

        # specific to cru
        # self.item_sensitivity_max = parameters["item_sensitivity_max"]
        # self.item_sensitivity_decrease = parameters["item_sensitivity_decrease"]
        self.encoding_drift_decrease = parameters["encoding_drift_decrease"]

        self.item_count = list_length
        self.encoding_probabilities = jnp.ones(self.item_count + 1)
        self.items = jnp.eye(self.item_count + 1)

        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context: Context = TemporalContext.init(list_length)
        self.mfc: Memory = LinearMemory.init_mfc(
            list_length + 1,
            self.context.size,
            parameters["learning_rate"],
            1.0,
        )
        self.mcf: Memory = LinearMemory.init_mcf(
            list_length + 1,
            self.context.size,
            parameters["item_support"],
            parameters["shared_support"],
            parameters["choice_sensitivity"],
        )
        self.decision_strategy = decision_strategy
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.zeros(self.item_count + 1, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """The learning rate for the MCF memory under its current state."""
        return self._mcf_learning_rate[self.study_index]

    @property
    def encoding_drift_rate(self) -> Float[Array, ""]:
        """The drift rate for encoding items."""
        return (
            self.encoding_drift_rate_max
            * self.encoding_drift_decrease**self.study_index
        )

    # @property
    # def item_sensitivity(self) -> Float[Array, ""]:
    #     """The sensitivity of items to encoding."""
    #     return (
    #         self.item_sensitivity_max * self.item_sensitivity_decrease**self.study_index
    #     )

    def experience_item(self, item_index: Int_) -> "CMR":
        """Return the model after experiencing item with the specified index.

        Args:
            item_index: the index of the item to experience. 0-indexed.
        """
        item = self.items[item_index]
        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)

        # item_supports = power_scale(
        #     letter_similarities[item_index] + lb, self.item_sensitivity
        # )
        item_encoding_probability = 1.0
        # self.decision_strategy.outcome_probability(
        #     item_index, item_supports
        # )

        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(item, new_context.state, self.mfc_learning_rate),
            mcf=self.mcf.associate(new_context.state, item, self.mcf_learning_rate),
            recallable=self.recallable.at[item_index].set(True),
            study_index=self.study_index + 1,
            encoding_probabilities=self.encoding_probabilities.at[item_index].set(
                item_encoding_probability
            ),
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
        #! add learning step for list termination "item" index
        termination_item = self.items[self.item_count]
        return self.replace(
            context=start_context,
            mfc=self.mfc.associate(termination_item, self.context.state, self.mfc_learning_rate),
            mcf=self.mcf.associate(self.context.state, termination_item, self.mcf_learning_rate),
            recallable=self.recallable.at[self.item_count].set(True),
        )

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of item with the specified index.

        Args:
            choice: the index of the item to retrieve (0-indexed)
        """
        new_context = self.context.integrate(
            self.mfc.probe(self.items[item_index]),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(item_index + 1),
            recallable=self.recallable.at[item_index].set(False),
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

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state"""
        item_activations = self.mcf.probe(self.context.state) + lb
        return item_activations * self.recallable  # mask recalled items

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state"""
        total_recallable = jnp.sum(self.recallable)
        return lax.cond(
            total_recallable == 0,
            true_fun=lambda: 1.0,
            false_fun=lambda: lax.cond(
                self.is_active,
                true_fun=lambda: jnp.minimum(
                    1.0 - (lb * total_recallable),
                    self.decision_strategy.outcome_probability(self.item_count, self.activations()),
                ),
                false_fun=lambda: 1.0,
            ),
        )

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Assumes that some items are recallable, with at least the minimum recall probability.

        Args:
            item_index: the index of the item to retrieve.
        """
        p_continue = 1 - self.stop_probability()
        item_activations = self.activations()
        probabilities = (
            self.encoding_probabilities
            * self.decision_strategy.outcome_probabilities(item_activations)
        )
        return p_continue * (probabilities / jnp.sum(probabilities))[item_index]

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
                p_stop == 1.0,
                lambda: 0.0,
                lambda: self.item_probability(choice - 1),
            ),
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return the outcome probabilities of all recall events."""
        p_stop = self.stop_probability()
        item_activations = self.activations()
        item_probabilities = (
            self.encoding_probabilities
            * self.decision_strategy.outcome_probabilities(item_activations)
        )
        return jnp.hstack(
            (
                p_stop,
                ((1 - p_stop) * (item_probabilities / jnp.sum(item_probabilities))),
            )
        )


class RaceDiffusionModel(Pytree):
    def outcome_probability(
        self,
        item_index: Int_,
        supports: Float[Array, " items"],
    ) -> Float_:
        """Returns the probability of selecting the specified item."""
        dt, pdfs, cdfs = race_diffusion_precompute(supports)
        return compute_runner_probability(item_index, pdfs, cdfs, dt)

    def outcome_probabilities(
        self,
        supports: Float[Array, " items"],
    ) -> Float[Array, " items"]:
        """Returns a probability distribution over all items."""
        dt, pdfs, cdfs = race_diffusion_precompute(supports)
        return compute_runner_probabilities(pdfs, cdfs, dt)


class FlatChoiceModel(Pytree):
    def outcome_probability(
        self,
        item_index: Int_,
        supports: Float[Array, " items"],
    ) -> Float_:
        """Returns the probability of selecting the specified item."""
        return supports[item_index] / jnp.sum(supports)

    def outcome_probabilities(
        self,
        supports: Float[Array, " items"],
    ) -> Float[Array, " items"]:
        """Returns a probability distribution over all items."""
        return supports / jnp.sum(supports)


class CMRFactory:
    def __init__(
        self,
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()

    def create_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return CMR(self.max_list_length, parameters, FlatChoiceModel())


class RacingCMRFactory:
    def __init__(
        self,
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()

    def create_model(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return CMR(self.max_list_length, parameters, RaceDiffusionModel())
