from jax import lax, numpy as jnp, jit
from typing import Callable, Mapping, Optional, Type, Iterable
import numpy as np
from cmr_mlx.typing import (
    MemorySearch,
    MemorySearchModelFactory,
    Integer,
    Array,
    Float_,
    Float,
    Real,
)

def all_rows_identical(arr: Real[Array, " x y"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0])  # type: ignore


def log_likelihood(likelihoods: Float[Array, "trial_count ..."]) -> Float[Array, ""]:
    """Return the summed log likelihood over specified likelihoods."""
    return -jnp.sum(jnp.log(likelihoods))


def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """Return the updated model and the outcome probabilities of a chain of retrieval events.

    Args:
        model: the current memory search model.
        choices: the indices of the items to retrieve (1-indexed) or 0 to stop.
    """
    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


class MemorySearchLikelihoodFnGenerator:
    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: dict[str, Integer[Array, " trials ?"]],
        connections: Optional[Integer[Array, " word_pool_items word_pool_items"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.factory = model_factory(dataset, connections)
        self.create_model = self.factory.create_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed
        self.trials = jnp.array(trials)

    def init_model_for_retrieval(self, trial_index, parameters):
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
    ) -> Integer[Array, " trials recall_events"]:
        """
        Predict outcomes for each trial using a single initial model
        if all present-lists match.
        Skips re-experiencing items for each trial.
        """
        model = self.init_model_for_retrieval(0, parameters)
        return lax.map(
            lambda i: predict_and_simulate_recalls(model, self.trials[i])[1],
            trial_indices,
        )

    def present_and_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Integer[Array, " trials recall_events"]:
        """
        Predict outcomes for each trial using a new model for each trial.
        Re-experiences items for each trial.
        """
        return lax.map(
            lambda i: predict_and_simulate_recalls(
                self.init_model_for_retrieval(i, parameters), self.trials[i]
            )[1],
            trial_indices,
        )

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        return log_likelihood(self.base_predict_trials(trial_indices, parameters))

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        return log_likelihood(
            self.present_and_predict_trials(trial_indices, parameters)
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_params: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        if all_rows_identical(self.present_lists[trial_indices]):
            base_loss_fn = self.base_predict_trials_loss
        else:
            base_loss_fn = self.present_and_predict_trials_loss

        def specialized_loss_fn(
            params: Mapping[str, Float_],
        ) -> Float[Array, ""]:
            return base_loss_fn(trial_indices, {**base_params, **params})

        @jit
        def single_param_loss(x):
            return specialized_loss_fn(
                {key: x[key_index] for key_index, key in enumerate(free_params)}
            )

        @jit
        def multi_param_loss(x):
            params = {
                key: jnp.array(x[key_index])
                for key_index, key in enumerate(free_params)
            }
            return lax.map(specialized_loss_fn, params)

        return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)
