"""Serial-position-curve MSE loss for memory-search models.

Computes an average squared error between observed and simulated serial position
curves by simulating recall chains per trial and aggregating the results with
``fixed_pres_spc``.

"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
from jax import numpy as jnp

from jaxcmr.analyses.spc import fixed_pres_spc
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
    RecallDataset,
)


__all__ = [
    "simulate_masked_free_recall",
    "MemorySearchSpcMseFnGenerator",
]

def simulate_masked_free_recall(
    model: MemorySearch,
    list_length: int,
    mask: jnp.ndarray,
    keys: PRNGKeyArray,
) -> Integer[Array, " recall_length"]:
    """Simulate free recall once and zero-out slots beyond the observed recall total.
    
    Args:
      model: Retrieval-ready memory search model.
      list_length: Length of the study list.
      mask: Boolean mask indicating valid recall positions.
      key: PRNG key for simulation.
    """
    def simulate_once(key: PRNGKeyArray) -> Integer[Array, " recall_length"]:
        _, recalls = simulate_free_recall(model, list_length, key)
        return recalls * mask
    return vmap(simulate_once)(keys)

class MemorySearchSpcMseFnGenerator:
    """Generates SPC-based MSE loss functions for a dataset and model factory."""

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
        dataset_recalls = jnp.array(dataset["recalls"], dtype=jnp.int32)
        self.simulation_count = 20

        trial_count = int(dataset_recalls.shape[0])
        base_key = random.PRNGKey(0)
        shared_key, per_trial_key = random.split(base_key)
        self.shared_simulation_keys = random.split(shared_key, self.simulation_count)
        per_trial_subkeys = random.split(per_trial_key, trial_count)
        self.simulation_keys = vmap(
            lambda key: random.split(key, self.simulation_count)
        )(per_trial_subkeys)

        self.trial_position_counts = vmap(
            lambda trial: jnp.bincount(trial, length=self.list_length + 1)[1:]
        )(dataset_recalls).astype(jnp.float32)
        self.trial_totals = self.trial_position_counts.sum(axis=1).astype(jnp.int32)
        prefix = jnp.arange(self.list_length, dtype=jnp.int32)
        self.trial_masks = prefix[None, :] < self.trial_totals[:, None]


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

    def base_spc_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns SPC MSE for identical study lists using shared simulations."""
        counts = self.trial_position_counts[trial_indices].sum(axis=0)
        target_spc = counts / trial_indices.size
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        simulated = vmap(simulate_masked_free_recall, in_axes=(None, None, 0, None))(
            model, self.list_length, self.trial_masks[trial_indices], self.shared_simulation_keys
        ).reshape(-1, self.list_length)
        simulated_spc = fixed_pres_spc(simulated, self.list_length)
        return jnp.mean(jnp.square(simulated_spc - target_spc))

    def present_and_predict_spc_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns SPC mean-squared error for the specified trials.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        counts = self.trial_position_counts[trial_indices].sum(axis=0)
        target_spc = counts / trial_indices.size
        models = vmap(self.init_model_for_retrieval, in_axes=(0, None))(
            trial_indices, parameters
        )
        simulated = vmap(simulate_masked_free_recall, in_axes=(0, None, 0, 0))(
            models, self.list_length, self.trial_masks[trial_indices], self.simulation_keys[trial_indices]
        ).reshape(-1, self.list_length)
        simulated_spc = fixed_pres_spc(simulated, self.list_length)
        return jnp.mean(jnp.square(simulated_spc - target_spc))

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
            base_loss_fn = self.base_spc_mse
        else:
            base_loss_fn = self.present_and_predict_spc_mse

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
