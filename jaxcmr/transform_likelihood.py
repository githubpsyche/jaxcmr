"""Loss generator variant with a likelihood-masking hook.

Applies a user-specified mask to per-trial likelihood arrays before computing
negative log-likelihood, enabling exclusion of selected recall events without
altering simulation behavior. Mask entries of 1 retain events; 0 neutralizes
them.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import all_rows_identical, log_likelihood
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Integer,
    LikelihoodMaskFn,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
)


def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """Returns updated model and event probabilities for a retrieval chain.

    Args:
      model: Current memory search model.
      choices: Retrieval indices (1-indexed) or 0 to stop.
    """
    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


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
    """Generates a loss function with a likelihood masking hook.

    The mask identifies recall events to retain before aggregation into
    negative log-likelihood. Mask values of False neutralize the event while
    keeping simulation behavior unchanged.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
        mask_likelihoods: LikelihoodMaskFn,
    ) -> None:
        """Initializes the generator with dataset, factory, and mask.

        Args:
          model_factory: Class implementing `MemorySearchModelFactory`.
          dataset: Recall dataset with presentations and recalls.
          connections: Optional connectivity matrix.
          mask_likelihoods: Function returning a boolean keep-mask for a recall vector.
        """
        self.factory = model_factory(dataset, connections)
        self.create_model = self.factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.mask_likelihoods = vmap(mask_likelihoods)
        self.has_connections = False if connections is None else jnp.any(connections).item()

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

    def init_model_for_retrieval(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Returns a retrieval-ready model for the trial's presentation list.

        Args:
          trial_index: Trial index to initialize.
          parameters: Model parameter mapping.
        """
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        return model.start_retrieving()

    def base_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " trials recall_events"]:
        """Returns event likelihoods using a single initialized model.

        Only valid when all presentation lists match (no re-presenting).

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameter mapping.
        """
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        return vmap(predict_and_simulate_recalls, in_axes=(None, 0))(
            model, self.trials[trial_indices]
        )[1]

    def present_and_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " trials recall_events"]:
        """Returns event likelihoods with a fresh model per trial.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameter mapping.
        """

        def present_and_predict_trial(i):
            model = self.init_model_for_retrieval(i, parameters)
            return predict_and_simulate_recalls(model, self.trials[i])[1]

        return vmap(present_and_predict_trial)(trial_indices)

    def _apply_mask(
        self,
        raw_likelihoods: Float[Array, " trials recall_events"],
        trial_indices: Integer[Array, " trials"],
    ) -> Float[Array, " trials recall_events"]:
        """Returns likelihoods with masked events neutralized.

        Mask entries set to True keep the original event likelihood; False
        entries are replaced with 1.0 to neutralize their contribution.

        Args:
          raw_likelihoods: Per-trial event likelihoods prior to masking.
          trial_indices: Trials associated with the likelihood rows.
        """
        recalls = self.trials[trial_indices]
        mask = self.mask_likelihoods(recalls)
        return raw_likelihoods * mask + (1.0 - mask)

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood for the base approach after masking.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameter mapping.
        """
        raw = self.base_predict_trials(trial_indices, parameters)
        masked = self._apply_mask(raw, trial_indices)
        return log_likelihood(masked)

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
        raw = self.present_and_predict_trials(trial_indices, parameters)
        masked = self._apply_mask(raw, trial_indices)
        return log_likelihood(masked)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to trials and free parameters.

        The returned function maps either a single parameter vector or a matrix of
        parameter vectors to negative log-likelihood values, applying the
        likelihood mask inside the loss paths.
        """
        # Decide which approach to use, based on whether all present-lists match
        if all_rows_identical(self.present_lists[trial_indices]) and not self.has_connections:
            base_loss_fn = self.base_predict_trials_loss
        else:
            base_loss_fn = self.present_and_predict_trials_loss

        def specialized_loss_fn(params: Mapping[str, Float_]) -> Float[Array, ""]:
            """Returns negative log-likelihood for a merged parameter mapping.

            Args:
              params: Free parameter mapping to merge with `base_params`.
            """
            return base_loss_fn(trial_indices, {**base_params, **params})

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            """Returns loss for one parameter vector."""
            param_dict = {key: x[i] for i, key in enumerate(free_param_names)}
            return specialized_loss_fn(param_dict)

        @jit
        def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
            """Returns one loss per parameter vector."""

            def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
                param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
                return specialized_loss_fn(param_dict)

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
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        self._inner = MemorySearchLikelihoodFnGenerator(
            model_factory,
            dataset,
            connections,
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
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        self._inner = MemorySearchLikelihoodFnGenerator(
            model_factory,
            dataset,
            connections,
            mask_likelihoods=mask_trailing_terminations,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)
