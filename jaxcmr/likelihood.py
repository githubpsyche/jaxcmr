"""Likelihood-based loss function generator for memory-search models.

Provides utilities to simulate retrieval event likelihoods per trial and to
aggregate them into a negative log-likelihood objective for fitting.
"""

from typing import Callable, Iterable, Mapping, Optional

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import all_rows_identical, log_likelihood
from jaxcmr.model_helpers import MemorySearchModelFactory
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    RecallDataset,
    MemorySearchCreateFn
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
    """Generates loss functions for a given dataset and model factory.

    Creates per-trial models, produces event likelihoods, and returns a callable
    that evaluates negative log-likelihood for parameter vectors.
    """
    def __init__(
        self,
        model_create_fn: MemorySearchCreateFn,
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize with dataset and optional feature embeddings.

        Args:
          model_create_fn: Function to create a memory search model.
          dataset: Trial-wise presentations and recalls.
          features: Optional feature matrix describing word-pool items.
        """
        self.factory = MemorySearchModelFactory(dataset, features, model_create_fn)
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

        self.trials = jnp.array(trials)

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

    def base_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " trials recall_events"]:
        """Returns event likelihoods using a single initialized model.

        Predicts all selected trials without re-presenting items (valid only when
        presentation lists are identical across the trials).

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
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
          parameters: Model parameters.
        """

        def present_and_predict_trial(i):
            model = self.init_model_for_retrieval(i, parameters)
            return predict_and_simulate_recalls(model, self.trials[i])[1]

        return vmap(present_and_predict_trial)(trial_indices)

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood for the base approach.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        return log_likelihood(self.base_predict_trials(trial_indices, parameters))

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log-likelihood for the present-and-predict approach.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        return log_likelihood(
            self.present_and_predict_trials(trial_indices, parameters)
        )

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
