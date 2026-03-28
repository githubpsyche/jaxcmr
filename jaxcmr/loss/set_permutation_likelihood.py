"""Set-permutation likelihood with masking, using per-trial study contexts.

Approximates recall-bag likelihoods by sampling permutations of observed
recall sets and evaluating their sequential probabilities under the model.
A user-specified mask controls which recall events contribute to the
negative log-likelihood.

"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
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
    PRNGKeyArray,
    RecallDataset,
)


__all__ = [
    "mask_trailing_terminations",
    "MemorySearchLikelihoodFnGenerator",
    "ExcludeTerminationLikelihoodFnGenerator",
    "IncludeTerminationLikelihoodFnGenerator",
]

def mask_trailing_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that retains nonzero recall events.

    Args:
      recalls: Recall indices for a single trial.
    """
    return jnp.not_equal(recalls, 0)


def _sample_permutations(
    trial_key: PRNGKeyArray,
    items: Integer[Array, " recall_events"],
    simulation_count: int,
) -> Integer[Array, " simulation_count recall_events"]:
    """Sample random permutations of a recall bag with zeros trailing.

    Shuffles the full item array ``simulation_count`` times, then
    stable-partitions each permutation so that all nonzero (recalled)
    items precede the zero-padding.  This guarantees that the model
    processes every recalled item before encountering any stop/padding
    events, which is required for the unconditional scan in
    ``predict_trial``.

    Parameters
    ----------
    trial_key : PRNGKeyArray
        Random key for this trial's permutations.
    items : Integer[Array, " recall_events"]
        Zero-padded recall bag (1-indexed item IDs, 0 for padding).
    simulation_count : int
        Number of permutations to sample.

    Returns
    -------
    Integer[Array, " simulation_count recall_events"]
        Permuted recall bags with zeros always trailing.

    """
    keys = random.split(trial_key, simulation_count)
    perms = vmap(lambda k: random.permutation(k, items))(keys)
    is_zero = (perms == 0).astype(jnp.int32)
    order = jnp.argsort(is_zero, axis=-1, stable=True)
    return jnp.take_along_axis(perms, order, axis=-1)


def _keep_all(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that retains all recall events."""
    return jnp.ones_like(recalls, dtype=bool)


class MemorySearchLikelihoodFnGenerator:
    """Masked set-permutation likelihood with per-trial study contexts.

    Approximates the recall-bag likelihood by averaging over random
    permutations of the observed recall set.  The mask controls which
    recall events contribute to the negative log-likelihood.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
        mask_likelihoods: LikelihoodMaskFn,
    ) -> None:
        """Initialize with dataset and optional feature embeddings.

        Args:
            model_factory: Class implementing `MemorySearchModelFactory`.
            dataset: Trial-wise presentations and recalls.
            features: Optional feature matrix describing word-pool items.
            mask_likelihoods: Function returning a boolean keep-mask for a recall vector.
        """
        self.create_model = model_factory(dataset, features).create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.simulation_count = 50
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
        trial_items = jnp.array(trials, dtype=jnp.int32)

        # Pre-sample permutations with zeros trailing
        trial_keys = random.split(self.base_key, trials.shape[0])
        self.trial_permutations = vmap(
            _sample_permutations, in_axes=(0, 0, None)
        )(trial_keys, trial_items, self.simulation_count)
        self.trial_masks = vmap(mask_likelihoods)(self.trial_permutations[:, 0])

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
        def _scan_trial(perm):
            return lax.scan(
                lambda m, c: (m.retrieve(c), m.outcome_probability(c)),
                model, perm,
            )[1]
        permutations = self.trial_permutations[trial_index]
        event_probs = vmap(_scan_trial)(permutations)

        # Apply pre-computed mask (same for all permutations since zeros trail)
        mask = self.trial_masks[trial_index]
        masked = event_probs * mask + (1.0 - mask)
        return jnp.mean(jnp.prod(masked, axis=-1))

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


class ExcludeTerminationLikelihoodFnGenerator:
    """Returns loss while ignoring trailing termination events.

    Trailing zeros in the recall matrix denote stop or padding events.
    Neutralizing them avoids penalizing lists for early stopping or
    padding conventions while keeping item recall events intact.
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


class IncludeTerminationLikelihoodFnGenerator:
    """Returns loss including stop event probabilities.

    All events contribute to the likelihood, including the first
    trailing zero (scored as the model's stop probability).  Subsequent
    zeros contribute 1.0 naturally because the model deactivates after
    the first stop.  Requires at least one padding slot per trial.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        # Validate: every trial needs at least one padding slot for stop
        recalled_counts = np.sum(np.array(dataset["recalls"]) > 0, axis=1)
        recall_width = np.array(dataset["recalls"]).shape[1]
        if np.any(recalled_counts >= recall_width):
            raise ValueError(
                "IncludeTerminationLikelihoodFnGenerator requires at least "
                "one padding slot per trial for the stop event, but one or "
                "more trials have no zero-padding in the recalls array."
            )
        self._inner = MemorySearchLikelihoodFnGenerator(
            model_factory,
            dataset,
            features,
            mask_likelihoods=_keep_all,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)
