"""Legacy likelihood generator that reuses a single study context.

This module mirrors the former ``base_predict`` shortcut used in other
likelihood generators. It assumes every trial shares the same presentation
sequence and therefore never replays item presentations per trial. The class is
useful for legacy experiments that relied on the old behaviour, but new code
should prefer the standard sequence likelihood which always re-presents the
study list.

"""

from typing import Callable, Iterable, Mapping, Type, Optional

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
    LikelihoodMaskFn,
)


__all__ = [
    "mask_trailing_terminations",
    "mask_first_recall",
    "predict_and_simulate_recalls",
    "MemorySearchLikelihoodFnGenerator",
    "ExcludeFirstRecallLikelihoodFnGenerator",
    "ExcludeTerminationLikelihoodFnGenerator",
]

def mask_trailing_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that retains nonzero recall events.

    Args:
      recalls: Recall indices for a single trial.
    """
    return jnp.not_equal(recalls, 0)


def mask_first_recall(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Returns keep-mask that drops the first recall event in a trial.

    Args:
      recalls: Recall indices for a single trial.
    """
    mask = jnp.ones_like(recalls, dtype=bool)
    return mask.at[0].set(False)


def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """Returns updated model and event probabilities for the provided choices."""

    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


class MemorySearchLikelihoodFnGenerator:
    """Likelihood generator that reuses a single initialized model among trials."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        mask_likelihoods: LikelihoodMaskFn,
    ) -> None:
        """Initializes the generator with dataset, factory, and mask.

        Args:
          model_factory: Class implementing `MemorySearchModelFactory`.
          dataset: Recall dataset with presentations and recalls.
          mask_likelihoods: Function returning a boolean keep-mask for a recall vector.
        """
        factory = model_factory(dataset, None)
        self.create_model = factory.create_trial_model
        self.presentation = jnp.array(dataset["pres_itemnos"])[0]
        self.mask_likelihoods = vmap(mask_likelihoods)

        # Reindex recalls so they align with ``presentation``.
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            recall = trials[trial_index]
            reindexed = np.array(
                [(self.presentation[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        self.trials = jnp.array(trials)

    def _apply_mask(
        self,
        raw_likelihoods: Float[Array, " trials recall_events"],
        recalls: Integer[Array, " trials recall_events"],
    ) -> Float[Array, " trials recall_events"]:
        """Returns likelihoods with masked events neutralized.

        Mask entries set to True keep the original event likelihood; False
        entries are replaced with 1.0 to neutralize their contribution.

        Args:
          raw_likelihoods: Per-trial event likelihoods prior to masking.
          recalls: Trials associated with the likelihood rows.
        """
        mask = self.mask_likelihoods(recalls)
        return raw_likelihoods * mask + (1.0 - mask)

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log likelihood across the selected trials."""
        model = self.create_model(0, parameters)
        model = lax.fori_loop(
            0,
            self.presentation.size,
            lambda i, m: m.experience(self.presentation[i]),
            model,
        )
        recalls = self.trials[trial_indices]
        raw = vmap(predict_and_simulate_recalls, in_axes=(None, 0))(
            model.start_retrieving(), recalls
        )[1]
        masked = self._apply_mask(raw, recalls)
        return log_likelihood(masked)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to the requested trials.

        Args:
            trial_indices: Trials to evaluate.
            base_params: Fixed model parameters.
            free_param_names: Names of parameters to optimize.
        """

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            param_dict = {key: x[i] for i, key in enumerate(free_param_names)}
            return self.base_predict_trials_loss(
                trial_indices, {**base_params, **param_dict}
            )

        @jit
        def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
            def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
                param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
                return self.base_predict_trials_loss(
                    trial_indices, {**base_params, **param_dict}
                )

            return vmap(loss_for_one_sample, in_axes=1)(x)

        return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)


class ExcludeFirstRecallLikelihoodFnGenerator:
    """Returns loss while ignoring the first recall event in each trial.

    Initializes a mask-enabled generator internally with a helper that
    neutralizes the first event.
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
            mask_likelihoods=mask_first_recall,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)


class ExcludeTerminationLikelihoodFnGenerator:
    """Returns loss while ignoring trailing termination events.

    Trailing zeros in the recall matrix denote explicit termination actions.
    Neutralizing them avoids penalizing lists for padding or early stopping
    conventions while keeping other recall events intact.
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
            mask_likelihoods=mask_trailing_terminations,
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        return self._inner(trial_indices, base_params, free_param_names)
