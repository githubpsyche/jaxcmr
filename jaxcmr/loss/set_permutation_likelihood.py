"""Likelihood-based loss function generator for memory-search models.

Approximates recall likelihoods by sampling permutations of observed recall bags
and evaluating their sequential probabilities under the model.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
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

        # Pre-sample permutations for each trial
        def sample_permutations(
            trial_key: PRNGKeyArray,
            items: Integer[Array, " recall_events"],
        ) -> Integer[Array, " simulation_count recall_events"]:
            keys = random.split(trial_key, self.simulation_count)
            return vmap(lambda k: random.permutation(k, items))(keys)

        trial_keys = random.split(self.base_key, trials.shape[0])
        self.trial_permutations: Integer[Array, "trials simulations list_length"] = (
            vmap(sample_permutations)(trial_keys, self.trial_items)
        )

    def predict_trial(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns Monte Carlo estimate of one trial's recall-bag probability.

        Args:
          trial_index: Trial index to simulate.
          parameters: Model parameters supplied to the factory.
        """
        # build and present to the model
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        model = model.start_retrieving()

        # predict across pre-sampled permutations
        permutations = self.trial_permutations[trial_index]
        event_probs = vmap(
            predict_and_simulate_recalls,
            in_axes=(None, 0),
        )(model, permutations)[1]
        return jnp.mean(jnp.prod(event_probs, axis=-1))

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
        raw = vmap(self.predict_trial, in_axes=(0, None))(trial_indices, parameters)
        return log_likelihood(raw)

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
