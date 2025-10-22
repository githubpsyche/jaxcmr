"""Category-filtered serial-position-curve MSE loss for memory-search models.

Computes mean squared error between observed and simulated category-specific
serial position curves by simulating recall chains per trial and aggregating the
results with ``fixed_pres_cat_spc``.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, random, vmap
from jax import numpy as jnp

from jaxcmr.analyses.cat_spc import fixed_pres_cat_spc, category_recall_counts
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


class MemorySearchCatSpcMseFnGenerator:
    """Generates category-filtered SPC-based MSE loss functions."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize with dataset, category metadata, and optional features.

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
        self.category_values = [1, 2]
        self.categories = jnp.array(dataset["Condition"], dtype=jnp.int32)

        trial_count = int(dataset_recalls.shape[0])
        base_key = random.PRNGKey(0)
        shared_key, per_trial_key = random.split(base_key)
        self.shared_simulation_keys = random.split(shared_key, self.simulation_count)
        per_trial_subkeys = random.split(per_trial_key, trial_count)
        self.simulation_keys = vmap(
            lambda key: random.split(key, self.simulation_count)
        )(per_trial_subkeys)

        self.trial_category_counts = jnp.stack(
            [
                vmap(
                    category_recall_counts,
                    in_axes=(0, 0, None, None),
                )(
                    dataset_recalls,
                    self.categories,
                    category_value,
                    self.list_length,
                ).astype(jnp.float32)
                for category_value in self.category_values
            ],
            axis=0,
        )
        self.trial_category_masks = jnp.stack(
            [
                (self.categories == category_value).astype(jnp.float32)
                for category_value in self.category_values
            ],
            axis=0,
        )
        self.trial_totals = jnp.count_nonzero(dataset_recalls, axis=1).astype(jnp.int32)
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

    def _target_cat_spc(
        self, trial_indices: Integer[Array, " trials"]
    ) -> Float[Array, " categories study_positions"]:
        """Returns category-filtered SPC target for selected trials."""
        counts = self.trial_category_counts[:, trial_indices].sum(axis=1)
        denominator = self.trial_category_masks[:, trial_indices].sum(axis=1)
        return counts / denominator #jnp.where(denominator > 0, counts / denominator, 0.0)

    def _simulated_cat_spc(
        self,
        recalls: Integer[Array, " trial_count simulation_count recall_length"],
        categories: Integer[Array, " trial_count study_positions"],
    ) -> Float[Array, " categories study_positions"]:
        """Returns category-filtered SPC for simulated recalls."""
        flattened_recalls = recalls.reshape(-1, recalls.shape[-1])
        repeated_categories = jnp.repeat(categories, self.simulation_count, axis=0)
        return jnp.stack(
            [
                fixed_pres_cat_spc(
                    flattened_recalls,
                    repeated_categories,
                    category_value,
                    self.list_length,
                )
                for category_value in self.category_values
            ],
            axis=0,
        )

    def base_spc_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns category SPC MSE using shared simulations."""
        target_cat_spc = self._target_cat_spc(trial_indices)
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        simulated = vmap(simulate_masked_free_recall, in_axes=(None, None, 0, None))(
            model,
            self.list_length,
            self.trial_masks[trial_indices],
            self.shared_simulation_keys,
        )
        categories = self.categories[trial_indices]
        simulated_cat_spc = self._simulated_cat_spc(simulated, categories)
        return jnp.mean(
            jnp.square(simulated_cat_spc.reshape(-1) - target_cat_spc.reshape(-1))
        )

    def present_and_predict_spc_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns category SPC mean-squared error for the specified trials.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameters.
        """
        target_cat_spc = self._target_cat_spc(trial_indices)
        models = vmap(self.init_model_for_retrieval, in_axes=(0, None))(
            trial_indices, parameters
        )
        simulated = vmap(simulate_masked_free_recall, in_axes=(0, None, 0, 0))(
            models,
            self.list_length,
            self.trial_masks[trial_indices],
            self.simulation_keys[trial_indices],
        )
        categories = self.categories[trial_indices]
        simulated_cat_spc = self._simulated_cat_spc(simulated, categories)
        return jnp.mean(
            jnp.square(simulated_cat_spc.reshape(-1) - target_cat_spc.reshape(-1))
        )

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
