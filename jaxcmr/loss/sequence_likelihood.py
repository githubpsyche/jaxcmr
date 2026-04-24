"""Unmasked sequential likelihood with per-trial study contexts.

Scores recall sequences in observed order, re-presenting the study list
for each trial.  All events — including trailing zeros (stop/padding) —
contribute to the negative log-likelihood.

"""

from typing import Iterable, Mapping, Optional, Type

import numpy as np
from jax import lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearchModelFactory,
    RecallDataset,
)


__all__ = [
    "MemorySearchLikelihoodLoss",
]


class MemorySearchLikelihoodLoss:
    """Unmasked sequential likelihood with per-trial study contexts.

    Re-presents the study list for each trial before scoring the recall
    sequence.  All events contribute to the negative log-likelihood,
    including trailing zeros (stop and padding events).
    """

    def __init__(
        self,
        model_factory_cls: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize with dataset and optional feature embeddings.

        Args:
          model_factory_cls: Class implementing `MemorySearchModelFactory`.
          dataset: Trial-wise presentations and recalls.
          features: Optional feature matrix describing word-pool items.
        """
        self.create_model = model_factory_cls(dataset, features).create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])

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

    def predict_trial(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " recall_events"]:
        """Returns recall-event probabilities for one trial.

        Args:
          trial_index: Index identifying which trial to simulate.
          parameters: Model parameters specific to the trial simulation.
        """
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        model = model.start_retrieving()
        return lax.scan(
            lambda m, c: (m.retrieve(c), m.outcome_probability(c)),
            model,
            self.trials[trial_index],
        )[1]

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log likelihood across specified trials."""

        raw = vmap(self.predict_trial, in_axes=(0, None))(trial_indices, parameters)
        return log_likelihood(raw)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: jnp.ndarray,
    ) -> Float[Array, " n_samples"]:
        """Returns one loss per parameter vector."""
        free_param_names = tuple(free_param_names)

        def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
            param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
            return self.present_and_predict_trials_loss(
                trial_indices, {**base_params, **param_dict}
            )

        return vmap(loss_for_one_sample, in_axes=1)(x)


# Compatibility aliases. Do not use in new code.
MemorySearchLikelihoodFnGenerator = MemorySearchLikelihoodLoss
