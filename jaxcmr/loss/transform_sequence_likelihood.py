"""Sequential likelihood with masking, using per-trial study contexts.

Scores recall sequences in observed order, re-presenting the study list
for each trial so that trial-specific features (e.g. EEG, item order)
are captured.  A user-specified mask controls which recall events
contribute to the negative log-likelihood.

"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Integer,
    LikelihoodMaskFn,
    MemorySearchModelFactory,
    RecallDataset,
)

__all__ = [
    "mask_trailing_terminations",
    "mask_first_recall",
    "MemorySearchLikelihoodFnGenerator",
    "ExcludeFirstRecallLikelihoodFnGenerator",
    "ExcludeTerminationLikelihoodFnGenerator",
]


def mask_trailing_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that retains nonzero recall events.

    Args:
      recalls: Recall indices for a single trial.
    """
    return jnp.not_equal(recalls, 0)


def mask_first_recall(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that drops the first recall event in a trial.

    Args:
      recalls: Recall indices for a single trial.
    """
    mask = jnp.ones_like(recalls, dtype=bool)
    return mask.at[0].set(False)


class MemorySearchLikelihoodFnGenerator:
    """Masked sequential likelihood with per-trial study contexts.

    Re-presents the study list for each trial before scoring the recall
    sequence, so trial-specific features are captured.  The mask
    controls which recall events contribute to the negative
    log-likelihood.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
        mask_likelihoods: LikelihoodMaskFn,
    ) -> None:
        """Initializes the generator with dataset, factory, and mask.

        Args:
            model_factory: Class implementing `MemorySearchModelFactory`.
            dataset: Recall dataset with presentations and recalls.
            features: Optional feature matrix for word-pool items.
            mask_likelihoods: Function returning a boolean keep-mask for a recall vector.
        """
        self.create_model = model_factory(dataset, features).create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        
        # Reindex the recalled items so they match the "present_lists" indexing
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        self.trials = jnp.array(trials)
        self.trial_masks = vmap(mask_likelihoods)(self.trials)

    def predict_trial(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " recall_events"]:
        """Returns recall-event probabilities for one trial.

        Args:
          trial_index: Index identifying which trial to simulate.
          parameters: Model parameters specific to the trial simulation.
        """
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        model = model.start_retrieving()
        return lax.scan(
            lambda m, c: (m.retrieve(c), m.outcome_probability(c)),
            model,
            self.trials[trial_index],
        )[1]

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood with per-trial models after masking.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameter mapping.
        """
        raw = vmap(self.predict_trial, in_axes=(0, None))(trial_indices, parameters)
        mask = self.trial_masks[trial_indices]
        masked = raw * mask + (1.0 - mask)
        return log_likelihood(masked)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to trials and free parameters.

        The returned function accepts either one parameter vector or a matrix of
        parameter vectors and returns corresponding negative log-likelihood values.

        Args:
          trial_indices: Trials to evaluate.
          base_params: Fixed parameters.
          free_param_names: Names and order of free parameters.
        """

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            """Returns loss for one parameter vector."""
            param_dict = {key: x[i] for i, key in enumerate(free_param_names)}
            return self.present_and_predict_trials_loss(
                trial_indices, {**base_params, **param_dict}
            )

        @jit
        def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
            """Returns one loss per parameter vector."""

            def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
                param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
                return self.present_and_predict_trials_loss(
                    trial_indices, {**base_params, **param_dict}
                )

            # vmap applies loss_for_one_sample across the leading dimension of x
            return vmap(loss_for_one_sample, in_axes=1)(x)

        # Return a function that checks the dimensionality of x at runtime
        return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)


class ExcludeFirstRecallLikelihoodFnGenerator:
    """Returns loss while ignoring the first recall event in each trial.

    Initializes a mask-enabled generator internally with a helper that
    neutralizes the first event.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        self._inner = MemorySearchLikelihoodFnGenerator(
            model_factory,
            dataset,
            features,
            mask_likelihoods=mask_first_recall,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)


class ExcludeTerminationLikelihoodFnGenerator:
    """Returns loss while ignoring trailing termination events.

    Trailing zeros in the recall matrix denote explicit termination actions.
    Neutralizing them avoids penalizing lists for padding or early stopping
    conventions while keeping other recall events intact.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        self._inner = MemorySearchLikelihoodFnGenerator(
            model_factory,
            dataset,
            features,
            mask_likelihoods=mask_trailing_terminations,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)
