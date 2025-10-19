"""Summed-permutation loss generator for order-free recall data."""

import itertools
from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from jaxcmr.helpers import all_rows_identical
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Integer,
    MemorySearch,
    RecallDataset,
    MemorySearchModelFactory,
)


def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """Returns updated model and event probabilities.

    Args:
      model: Current memory search model.
      choices: Retrieval indices (1-indexed) or 0 to stop.
    """
    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


class MemorySearchLikelihoodFnGenerator:
    """Generates loss functions that sum likelihoods across recall permutations."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize with dataset and optional feature embeddings.

        Args:
          model_factory: Class implementing `MemorySearchModelFactory`.
          dataset: Trial-wise presentations and recalls.
          features: Optional feature matrix describing word-pool items.
        """
        self.factory = model_factory(dataset, features)
        self.create_model = self.factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.has_features = False if features is None else jnp.any(features).item()

        # Reindex the recalled items so they match the "present_lists" indexing
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        permutation_rows: list[np.ndarray] = []
        permutation_counts: list[int] = []
        max_permutations = 1

        for recall in trials:
            positives = recall[recall > 0]
            if positives.size == 0:
                padded = np.zeros((1, recall.shape[0]), dtype=int)
            else:
                unique = list(
                    dict.fromkeys(itertools.permutations(positives.tolist()))
                )
                perms = np.asarray(unique, dtype=int)
                padded = np.zeros((perms.shape[0], recall.shape[0]), dtype=int)
                padded[:, : positives.size] = perms
            permutation_rows.append(padded)
            permutation_counts.append(padded.shape[0])
            max_permutations = max(max_permutations, padded.shape[0])

        permutation_tensor = np.zeros(
            (trials.shape[0], max_permutations, trials.shape[1]), dtype=int
        )
        for idx, padded in enumerate(permutation_rows):
            permutation_tensor[idx, : padded.shape[0], :] = padded

        self.permutations = jnp.array(permutation_tensor)
        self.permutation_counts = jnp.array(permutation_counts, dtype=jnp.int32)
        self.max_permutations = int(max_permutations)

    def init_model_for_retrieval(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Returns a retrieval-ready model for a trial.

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

    def _sequence_log_probabilities(
        self,
        model: MemorySearch,
        sequences: Integer[Array, " permutations recall_events"],
    ) -> Float[Array, " permutations"]:
        """Returns log probabilities for each recall sequence."""
        _, probabilities = vmap(predict_and_simulate_recalls, in_axes=(None, 0))(
            model, sequences
        )
        return jnp.sum(jnp.log(probabilities), axis=1)

    def _mask_permutations(
        self, counts: Integer[Array, " trials"]
    ) -> Bool[Array, " trials permutations"]:
        """Returns mask indicating valid permutations per trial."""
        return jnp.arange(self.max_permutations) < counts[:, None]

    def base_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " trials"]:
        """Returns log-likelihoods summed across permutations for each trial."""
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        perms = self.permutations[trial_indices]
        counts = self.permutation_counts[trial_indices]
        flat_perms = perms.reshape(-1, perms.shape[-1])
        log_probs = self._sequence_log_probabilities(model, flat_perms)
        mask = self._mask_permutations(counts).reshape(-1)
        log_probs = jnp.where(mask, log_probs, -jnp.inf)
        log_probs = log_probs.reshape(perms.shape[0], self.max_permutations)
        return logsumexp(log_probs, axis=1)

    def present_and_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " trials"]:
        """Returns log-likelihoods summed across permutations with per-trial models."""

        def per_trial(i):
            model = self.init_model_for_retrieval(i, parameters)
            perms = self.permutations[i]
            log_probs = self._sequence_log_probabilities(model, perms)
            mask = jnp.arange(self.max_permutations) < self.permutation_counts[i]
            log_probs = jnp.where(mask, log_probs, -jnp.inf)
            return logsumexp(log_probs)

        return vmap(per_trial)(trial_indices)

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood for the base approach."""

        return -jnp.sum(self.base_predict_trials(trial_indices, parameters))

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood for the present-and-predict approach."""
        return -jnp.sum(self.present_and_predict_trials(trial_indices, parameters))

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
        # Decide which approach to use, based on whether all present-lists match
        if all_rows_identical(self.present_lists[trial_indices]) and not self.has_features:
            base_loss_fn = self.base_predict_trials_loss
        else:
            base_loss_fn = self.present_and_predict_trials_loss

        def specialized_loss_fn(params: Mapping[str, Float_]) -> Float[Array, ""]:
            """Returns negative log-likelihood for merged base and free params."""
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
