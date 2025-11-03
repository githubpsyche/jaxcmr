"""Legacy likelihood generator that reuses a single study context.

This module mirrors the former ``base_predict`` shortcut used in other
likelihood generators. It assumes every trial shares the same presentation
sequence and therefore never replays item presentations per trial. The class is
useful for legacy experiments that relied on the old behaviour, but new code
should prefer the standard sequence likelihood which always re-presents the
study list.
"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
)


def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """Returns updated model and event probabilities for the provided choices."""

    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


class BaseSequenceLikelihoodFnGenerator:
    """Likelihood generator that reuses a single initialized model among trials."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        factory = model_factory(dataset, features)
        self.create_model = factory.create_trial_model
        self.presentation = jnp.array(dataset["pres_itemnos"])[0]

        # Reindex recalls so they align with ``presentation``.
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            recall = trials[trial_index]
            reindexed = np.array(
                [(self.presentation[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        self.trials = jnp.array(trials)

    def base_predict_trials_loss(
        self,
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log likelihood across all trials using the shared model."""
        model = self.create_model(0, parameters)
        model = lax.fori_loop(
            0,
            self.presentation.size,
            lambda i, m: m.experience(self.presentation[i]),
            model,
        )
        raw = vmap(predict_and_simulate_recalls, in_axes=(None, 0))(
            model.start_retrieving(), self.trials
        )[1]
        return log_likelihood(raw)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to the requested trials.

        Args:
            trial_indices: Trials to evaluate (ignored).
            base_params: Fixed model parameters.
            free_param_names: Names of parameters to optimize.
        """

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            param_dict = {key: x[i] for i, key in enumerate(free_param_names)}
            return self.base_predict_trials_loss({**base_params, **param_dict})

        @jit
        def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
            def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
                param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
                return self.base_predict_trials_loss({**base_params, **param_dict})

            return vmap(loss_for_one_sample, in_axes=1)(x)

        return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)
