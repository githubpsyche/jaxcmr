"""Mean-squared-error loss for recall analyses.

Provides dataset-aware helpers to compare recall-derived analyses between
observed and simulated trials using pluggable analysis functions.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
from jax import numpy as jnp

from jaxcmr.helpers import all_rows_identical
from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallAnalysisFn,
    RecallDataset,
)

__all__ = [
    "simulate_masked_free_recall",
    "MemorySearchMseFnGenerator",
]

def simulate_masked_free_recall(
    model: MemorySearch,
    list_length: int,
    mask: jnp.ndarray,
    keys: PRNGKeyArray,
) -> Integer[Array, "simulation_count recall_length"]:
    """Simulate recall sequences and zero masked slots.

    Args:
      model: Retrieval-ready memory search model.
      list_length: Study list length.
      mask: Boolean mask marking observed recall slots.
      keys: PRNG keys for repeated simulations.
    """

    def simulate_once(key: PRNGKeyArray) -> Integer[Array, " recall_length"]:
        _, recalls = simulate_free_recall(model, list_length, key)
        return recalls * mask

    return vmap(simulate_once)(keys)


class MemorySearchMseFnGenerator:
    """Generate mean-squared-error losses for recall analyses."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
        analysis_fn: RecallAnalysisFn,
        simulation_count: int = 20,
    ) -> None:
        """Configure dataset-specific loss generation.

        Args:
          model_factory: Class implementing `MemorySearchModelFactory`.
          dataset: Trial-wise presentations and recalls.
          features: Optional feature matrix describing word-pool items.
          analysis_fn: Callable that summarizes recall data for comparison.
          simulation_count: Number of simulated recall chains per trial.
        """
        # Configure factories and analysis hooks.
        factory = model_factory(dataset, features)
        self.create_model = factory.create_trial_model
        self.analysis_fn = analysis_fn
        self.simulation_count = simulation_count

        # Cache dataset fields as JAX arrays for vectorized processing.
        self.dataset_view = {
            key: jnp.asarray(value)
            for key, value in dataset.items()
        }

        self.present_lists = jnp.asarray(
            self.dataset_view["pres_itemnos"], dtype=jnp.int32
        )
        self.dataset_view["pres_itemnos"] = self.present_lists

        self.list_length = int(self.present_lists.shape[1])

        recalls = jnp.asarray(self.dataset_view["recalls"], dtype=jnp.int32)
        self.dataset_view["recalls"] = recalls
        self.dataset_recalls = recalls

        # Track whether per-trial feature embeddings are in use.
        self.has_features = False
        if features is not None:
            features_array = jnp.asarray(features)
            self.has_features = bool(jnp.any(features_array).item())

        # Pre-generate PRNG keys used for recall simulations.
        trial_count = int(recalls.shape[0])
        base_key = random.PRNGKey(0)
        shared_key, per_trial_key = random.split(base_key)
        self.shared_simulation_keys = random.split(shared_key, simulation_count)
        per_trial_subkeys = random.split(per_trial_key, trial_count)
        self.simulation_keys = vmap(
            lambda key: random.split(key, simulation_count)
        )(per_trial_subkeys)

        # Build boolean masks that gate simulated recalls to observed lengths.
        recall_totals = jnp.count_nonzero(recalls, axis=1).astype(jnp.int32)
        prefix = jnp.arange(self.list_length, dtype=jnp.int32)
        self.trial_masks = prefix[None, :] < recall_totals[:, None]

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

    def _target_analysis(
        self, trial_indices: Integer[Array, "trials"]
    ) -> Float[Array, "analysis_values"]:
        """Return analysis values for observed recalls."""
        observed = self.dataset_recalls[trial_indices][:, None, :]
        return self.analysis_fn(
            observed,
            list_length=self.list_length,
            dataset=self.dataset_view,
            trial_indices=trial_indices,
        )

    def base_analysis_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Return analysis MSE for identical lists using shared simulations.
        
        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        target = self._target_analysis(trial_indices)
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        simulated = vmap(
            simulate_masked_free_recall, in_axes=(None, None, 0, None)
        )(
            model,
            self.list_length,
            self.trial_masks[trial_indices],
            self.shared_simulation_keys,
        )
        simulated_values = self.analysis_fn(
            simulated,
            list_length=self.list_length,
            dataset=self.dataset_view,
            trial_indices=trial_indices,
        )
        return jnp.mean(jnp.square(simulated_values - target))

    def present_and_predict_analysis_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Return analysis MSE with individualized simulations.
        
        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        target = self._target_analysis(trial_indices)
        models = vmap(self.init_model_for_retrieval, in_axes=(0, None))(
            trial_indices, parameters
        )
        simulated = vmap(
            simulate_masked_free_recall, in_axes=(0, None, 0, 0)
        )(
            models,
            self.list_length,
            self.trial_masks[trial_indices],
            self.simulation_keys[trial_indices],
        )#.reshape(-1, self.list_length)
        simulated_values = self.analysis_fn(
            simulated,
            list_length=self.list_length,
            dataset=self.dataset_view,
            trial_indices=trial_indices,
        )
        return jnp.mean(jnp.square(simulated_values - target))

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to trials and free parameters.

        Args:
          trial_indices: Trials to evaluate.
          base_params: Fixed parameters.
          free_param_names: Names and order of free parameters.
        """

        if (
            all_rows_identical(self.present_lists[trial_indices])
            and not self.has_features
        ):
            base_loss_fn = self.base_analysis_mse
        else:
            base_loss_fn = self.present_and_predict_analysis_mse

        def specialized_loss_fn(params: Mapping[str, Float_]) -> Float[Array, ""]:
            """Returns loss for the merged base and free parameters."""
            return base_loss_fn(trial_indices, {**base_params, **params})

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            """Returns loss for a single parameter vector."""
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
