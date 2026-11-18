"""Score the observed order of recall under a memory-search model.

If a participant recalls C and then A, the model assigns a probability
to C after study and a probability to A after retrieving C.
``sequence_probabilities`` follows those same responses through a model
that is ready to retrieve. ``MemorySearchLikelihoodLoss`` first presents
each trial's study list and translates recorded study positions into the
model's item codes. It then sums the negative log probabilities of the
selected events across trials.

Use the named losses to omit the first response, termination, or both from
the score. Omitting C from the score in this example still conditions the
probability of A on C. A recorded zero represents STOP. Omitting its
contribution leaves the model's stopping rule active when it assigns
probabilities to items.
"""

from collections.abc import Iterable

import jax.numpy as jnp
from jax import lax, vmap

from jaxcmr.dataset import recall_positions_to_items
from jaxcmr.math import negative_log_likelihood
from jaxcmr.simulation import simulate_study
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Integer,
    LikelihoodMaskFn,
    MemorySearch,
    MemorySearchModelFactory,
    Parameters,
    RecallDataset,
)

__all__ = [
    "ExcludeFirstRecallAndTerminationLikelihoodLoss",
    "ExcludeFirstRecallLikelihoodLoss",
    "ExcludeTerminationLikelihoodLoss",
    "MemorySearchLikelihoodLoss",
    "mask_first_recall",
    "mask_first_recall_and_terminations",
    "mask_trailing_terminations",
    "sequence_probabilities",
]


def sequence_probabilities(
    model: MemorySearch,
    recalls: Integer[Array, " recall_events"],
) -> Float[Array, " recall_events"]:
    """Calculate each observed outcome's probability before retrieving it.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready model. Its initial state is not changed.
    recalls : Integer[Array, " recall_events"]
        Within-list item codes in observed order, including zero-coded STOP.

    Returns
    -------
    Float[Array, " recall_events"]
        One probability per supplied outcome, conditioned on all earlier ones.
        Scoring masks belong to the caller; they never remove conditioning events.
    """
    return lax.scan(
        lambda state, item: (state.retrieve(item), state.outcome_probability(item)),
        model,
        recalls,
    )[1]


def mask_trailing_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Keep nonzero events in a compact, zero-padded recall sequence."""
    return recalls != 0


def mask_first_recall(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Exclude the first recorded slot, retaining subsequent events and STOP."""
    return jnp.arange(recalls.size) != 0


def mask_first_recall_and_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Keep nonzero recall events after the first recorded slot."""
    return mask_first_recall(recalls) & mask_trailing_terminations(recalls)


class MemorySearchLikelihoodLoss:
    """Sum observed-sequence negative log likelihood over selected trials.

    Parameters
    ----------
    model_factory_cls : type[MemorySearchModelFactory]
        Factory preparing a model for each trial.
    dataset : RecallDataset
        Study lists and position-coded recalls, with trial-specific metadata.
    features : Array or None
        Vocabulary features passed to the factory.
    mask_likelihoods : LikelihoodMaskFn or None, optional
        Keep-mask function receiving each converted item sequence. None scores
        every supplied event. Masked events still advance the model.

    Notes
    -----
    Candidate matrices have parameters in rows and candidates in columns.
    Impossible scored events give infinite loss; masked probabilities are
    replaced with one before logarithms. No epsilon or probability clipping
    changes the likelihood. Calls support JIT and differentiation.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        mask_likelihoods: LikelihoodMaskFn | None = None,
    ) -> None:
        self.factory = model_factory_cls(dataset, features)
        self.create_model = self.factory.create_trial_model
        self.present_lists = jnp.asarray(dataset["pres_itemnos"])
        self.trials = recall_positions_to_items(dataset["recalls"], self.present_lists)
        self.trial_masks = (
            jnp.ones_like(self.trials, dtype=bool)
            if mask_likelihoods is None
            else vmap(mask_likelihoods)(self.trials)
        )

    def predict_trial(
        self, trial_index: Integer[Array, ""], parameters: Parameters
    ) -> Float[Array, " recall_events"]:
        """Return event probabilities after studying the selected trial's list."""
        model = self.create_model(trial_index, parameters)
        model = simulate_study(
            model, self.present_lists[trial_index]
        ).start_retrieving()
        return sequence_probabilities(model, self.trials[trial_index])

    def present_and_predict_trials_loss(
        self, trial_indices: Integer[Array, " trials"], parameters: Parameters
    ) -> Float[Array, ""]:
        """Return summed masked negative log likelihood for one parameter set."""
        probabilities = vmap(self.predict_trial, in_axes=(0, None))(
            trial_indices, parameters
        )
        return negative_log_likelihood(
            jnp.where(self.trial_masks[trial_indices], probabilities, 1.0)
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Parameters,
        free_param_names: Iterable[str],
        x: Float[Array, " parameters samples"],
    ) -> Float[Array, " samples"]:
        """Score candidate columns using masked sequence likelihood.
        
        Parameters
        ----------
        trial_indices : Integer[Array, "trials"]
            Dataset rows to score, preserving their order and multiplicity.
        base_params : Parameters
            Fixed model values and defaults overridden by candidate values.
        free_param_names : Iterable[str]
            Names in the order of the candidate matrix's parameter rows.
        x : Float[Array, "parameters samples"]
            Candidate columns in physical model units.
        
        Returns
        -------
        Float[Array, "samples"]
            One objective value per candidate column, in input order.
        """
        names = tuple(free_param_names)

        def score(values: Float[Array, " parameters"]) -> Float[Array, ""]:
            return self.present_and_predict_trials_loss(
                trial_indices, dict(base_params) | dict(zip(names, values))
            )

        return vmap(score, in_axes=1)(x)


class ExcludeFirstRecallLikelihoodLoss(MemorySearchLikelihoodLoss):
    """Score subsequent outcomes, conditioning on the unscored first recall."""

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        super().__init__(model_factory_cls, dataset, features, mask_first_recall)


class ExcludeTerminationLikelihoodLoss(MemorySearchLikelihoodLoss):
    """Score recalled items while excluding zero-coded STOP and padding."""

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        super().__init__(
            model_factory_cls, dataset, features, mask_trailing_terminations
        )


class ExcludeFirstRecallAndTerminationLikelihoodLoss(MemorySearchLikelihoodLoss):
    """Score items after the first recall, excluding STOP and padding."""

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        super().__init__(
            model_factory_cls, dataset, features, mask_first_recall_and_terminations
        )
