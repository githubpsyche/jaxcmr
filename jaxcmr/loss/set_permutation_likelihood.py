"""Likelihood-based loss function generator for memory-search models.

Approximates recall likelihoods by sampling permutations of observed recall bags
and evaluating their sequential probabilities under the model.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
from jax import numpy as jnp

from jaxcmr.helpers import all_rows_identical, log_likelihood
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallDataset,
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
        lambda m, c: lax.cond(
            c > 0,
            lambda: (m.retrieve(c), m.outcome_probability(c)),
            lambda: (m, jnp.array(1.0, dtype=jnp.float32)),
        ),
        model,
        choices,
    )


class MemorySearchLikelihoodFnGenerator:
    """Generates loss functions for a given dataset and model factory.

    Creates per-trial models, produces event likelihoods, and returns a callable
    that evaluates negative log-likelihood for parameter vectors.
    """

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
        factory = model_factory(dataset, features)
        self.create_model = factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.list_length = self.present_lists.shape[1]
        self.has_features = False if features is None else jnp.any(features).item()
        self.simulation_count = 40
        self.base_key = random.PRNGKey(0)

        # Reindex the recalled items so they match the "present_lists" indexing
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed
        trials = jnp.array(trials, dtype=jnp.int32)
        self.trial_items = trials

        # Store RNG scaffolding so we can sample permutations with static shapes
        shared_key, per_trial_key = random.split(self.base_key)
        self.shared_simulation_keys = random.split(shared_key, self.simulation_count)
        per_trial_subkeys = random.split(per_trial_key, trials.shape[0])
        self.simulation_keys = vmap(
            lambda key: random.split(key, self.simulation_count)
        )(per_trial_subkeys)

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

    def estimate_bag_probability(
        self,
        model: MemorySearch,
        trial_index: Integer[Array, ""],
        keys: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Returns Monte Carlo estimate of bag probability for a trial.

        Args:
          model: Initialized retrieval model for the trial.
          trial_index: Trial index to evaluate.
          keys: Simulation keys to use for the Monte Carlo estimate.
        """
        items = self.trial_items[trial_index]

        def permutation_probability(key: PRNGKeyArray) -> Float[Array, ""]:
            perm = random.permutation(key, items)
            _, event_probs = predict_and_simulate_recalls(model, perm)
            return jnp.prod(event_probs)

        return jnp.mean(
            vmap(permutation_probability)(keys)
        )

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns bag likelihoods using a single initialized model.

        Predicts all selected trials without re-presenting items (valid only when
        presentation lists are identical across the trials).

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        raw_probabilities = vmap(
            self.estimate_bag_probability, in_axes=(None, 0, None)
        )(model, trial_indices, self.shared_simulation_keys)
        return log_likelihood(raw_probabilities)

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

        def present_and_predict_trial(i):
            model = self.init_model_for_retrieval(i, parameters)
            keys = self.simulation_keys[i]
            return self.estimate_bag_probability(model, i, keys)

        raw_probabilities = vmap(present_and_predict_trial)(trial_indices)
        return log_likelihood(raw_probabilities)

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
        if (
            all_rows_identical(self.present_lists[trial_indices])
            and not self.has_features
        ):
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
