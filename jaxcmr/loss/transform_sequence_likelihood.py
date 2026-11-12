"""Sequential likelihood with masking, using per-trial study contexts.

Scores recall sequences in observed order, re-presenting the study list
for each trial so that trial-specific features (e.g. EEG, item order)
are captured.  A user-specified mask controls which recall events
contribute to the negative log-likelihood.

"""

from collections.abc import Iterable, Mapping

import numpy as np
from jax import lax, vmap
from jax import numpy as jnp
from jax.core import Tracer

from jaxcmr.helpers import log_likelihood
from jaxcmr.loss.prepared import PreparedLoss, prediction_likelihood
from jaxcmr.models.sequence import PreparedPrediction, make_sequential_evaluator
from jaxcmr.typing import (
    Array,
    BatchSequencePredictionFactory,
    Bool,
    Float,
    Float_,
    Integer,
    LikelihoodMaskFn,
    MemorySearchModelFactory,
    RecallDataset,
    SequencePredictionFactory,
)

__all__ = [
    "ExcludeFirstRecallLikelihoodLoss",
    "ExcludeFirstRecallAndTerminationLikelihoodLoss",
    "ExcludeTerminationLikelihoodLoss",
    "MemorySearchLikelihoodLoss",
    "mask_first_recall",
    "mask_first_recall_and_terminations",
    "mask_trailing_terminations",
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


def mask_first_recall_and_terminations(
    recalls: Integer[Array, " recall_events"],
) -> Bool[Array, " recall_events"]:
    """Keep nonzero recall events after the observed first recall.

    Args:
        recalls: Compact recall sequence with trailing zero padding.
    """
    return mask_first_recall(recalls) & mask_trailing_terminations(recalls)


class MemorySearchLikelihoodLoss:
    """Masked sequential likelihood with per-trial study contexts.

    Re-presents the study list for each trial before scoring the recall
    sequence, so trial-specific features are captured.  The mask
    controls which recall events contribute to the negative
    log-likelihood.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        mask_likelihoods: LikelihoodMaskFn,
    ) -> None:
        """Initializes the generator with dataset, factory, and mask.

        Args:
            model_factory_cls: Class implementing `MemorySearchModelFactory`.
            dataset: Recall dataset with presentations and recalls.
            features: Optional feature matrix for word-pool items.
            mask_likelihoods: Function returning a boolean keep-mask for a recall vector.
        """
        self.factory = model_factory_cls(dataset, features)
        self.create_model = self.factory.create_trial_model
        self._sequence_evaluator = make_sequential_evaluator(self.factory)
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
        self.trial_masks = vmap(mask_likelihoods)(self.trials)

    def _indices(self, trial_indices: Array) -> Array:
        """Validate concrete dataset rows for preparation outside JIT."""
        if isinstance(trial_indices, Tracer):
            raise TypeError(
                "Prepare predictions before JIT with concrete trial indices."
            )
        indices = np.asarray(trial_indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= len(self.trials)))
        ):
            raise ValueError(
                "trial_indices must be a one-dimensional array of valid trial indices."
            )
        return jnp.asarray(indices, dtype=jnp.int32)

    def prepare_trials(self, trial_indices: Array) -> PreparedLoss:
        """Ask the selected model for predictions; otherwise use its event methods.

        The model owns any bulk computation. This adapter only supplies scoring
        requests and reduces its returned probabilities. The direct __call__
        path remains the generic event calculation for dynamic trial indices.

        Args:
            trial_indices: Concrete one-dimensional integer array of dataset
                rows. Order and repeated indices are preserved.

        Returns:
            A callable accepting base parameters, free parameter names and a
            parameter-by-sample matrix, returning one loss per sample. Its
            observation arrays can be passed dynamically through JIT.
        """
        indices = self._indices(trial_indices)
        mask = self.trial_masks[indices]
        if isinstance(self.factory, SequencePredictionFactory):
            prediction = self.factory.prepare_predictions(indices, mask)
        else:
            # Selected arrays remain dynamic data; evaluator identity stays
            # fixed across selections, including separately prepared calls.
            prediction = PreparedPrediction(
                (indices, self.present_lists[indices], self.trials[indices]),
                (mask,),
                self._sequence_evaluator,
            )
        return PreparedLoss(prediction, prediction_likelihood)

    def prepare_selections(
        self, trial_indices: Iterable[Array]
    ) -> tuple[PreparedLoss, ...]:
        """Prepare selections, delegating common array layouts to the factory.

        Args:
            trial_indices: One concrete trial-index array per selection, such
                as one array per participant.

        Returns:
            One prepared loss per selection in input order. When the factory
            supports it, compatible selections share shapes and compiled code.
            The ordinary fallback also shares compiled code whenever selected
            arrays already have matching shapes and dtypes.
        """
        selections = tuple(self._indices(indices) for indices in trial_indices)
        if isinstance(self.factory, BatchSequencePredictionFactory):
            masks = tuple(self.trial_masks[indices] for indices in selections)
            predictions = self.factory.prepare_prediction_selections(selections, masks)
            if len(predictions) != len(selections):
                raise ValueError("The model must return one prediction per selection.")
            return tuple(
                PreparedLoss(prediction, prediction_likelihood)
                for prediction in predictions
            )
        return tuple(self.prepare_trials(indices) for indices in selections)

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
        """Returns negative log-likelihood with per-trial models after masking.

        Args:
          trial_indices: Trials to evaluate.
          parameters: Model parameter mapping.
        """
        raw = vmap(self.predict_trial, in_axes=(0, None))(trial_indices, parameters)
        mask = self.trial_masks[trial_indices]
        masked = raw * mask + (1.0 - mask)
        return log_likelihood(masked)

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


class ExcludeFirstRecallLikelihoodLoss:
    """Returns loss while ignoring the first recall event in each trial.

    Initializes a mask-enabled generator internally with a helper that
    neutralizes the first event.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        self._inner = MemorySearchLikelihoodLoss(
            model_factory_cls,
            dataset,
            features,
            mask_likelihoods=mask_first_recall,
        )

    def prepare_trials(self, trial_indices: Array) -> PreparedLoss:
        """Prepare selected rows, conditioning on the unscored first recall.

        Args:
            trial_indices: Concrete one-dimensional array of dataset rows.

        Returns:
            Prepared loss using the same first-recall mask as direct evaluation.
        """
        return self._inner.prepare_trials(trial_indices)

    def prepare_selections(
        self, trial_indices: Iterable[Array]
    ) -> tuple[PreparedLoss, ...]:
        """Prepare trial-index selections in order, sharing layouts when supported.

        Args:
            trial_indices: One concrete dataset-row array per selection.

        Returns:
            One prepared loss per selection, excluding first-recall scores.
        """
        return self._inner.prepare_selections(trial_indices)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: jnp.ndarray,
    ) -> Float[Array, " n_samples"]:
        return self._inner(trial_indices, base_params, free_param_names, x)


class ExcludeTerminationLikelihoodLoss:
    """Returns loss while ignoring trailing termination events.

    Trailing zeros in the recall matrix denote explicit termination actions.
    Neutralizing them avoids penalizing lists for padding or early stopping
    conventions while keeping other recall events intact.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        self._inner = MemorySearchLikelihoodLoss(
            model_factory_cls,
            dataset,
            features,
            mask_likelihoods=mask_trailing_terminations,
        )

    def prepare_trials(self, trial_indices: Array) -> PreparedLoss:
        """Prepare selected rows with zero-coded termination events unscored.

        Args:
            trial_indices: Concrete one-dimensional array of dataset rows.

        Returns:
            Prepared loss using the same termination mask as direct evaluation.
        """
        return self._inner.prepare_trials(trial_indices)

    def prepare_selections(
        self, trial_indices: Iterable[Array]
    ) -> tuple[PreparedLoss, ...]:
        """Prepare trial-index selections in order, sharing layouts when supported.

        Args:
            trial_indices: One concrete dataset-row array per selection.

        Returns:
            One prepared loss per selection, excluding termination scores.
        """
        return self._inner.prepare_selections(trial_indices)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: jnp.ndarray,
    ) -> Float[Array, " n_samples"]:
        return self._inner(trial_indices, base_params, free_param_names, x)


class ExcludeFirstRecallAndTerminationLikelihoodLoss:
    """Score later recalls conditional on the observed first recall, without STOP.

    The first recall still updates the model before later choices are scored.
    Pair this loss with NoStopTermination to exclude stopping probabilities
    from the nonzero choice probabilities as well. Keep the full compact
    recall sequence in the dataset; only scoring is masked.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
    ) -> None:
        self._inner = MemorySearchLikelihoodLoss(
            model_factory_cls,
            dataset,
            features,
            mask_likelihoods=mask_first_recall_and_terminations,
        )

    def prepare_trials(self, trial_indices: Array) -> PreparedLoss:
        """Prepare selected rows, conditioning on the unscored first recall.

        Args:
            trial_indices: Concrete one-dimensional array of dataset rows.

        Returns:
            Prepared loss using the same combined mask as direct evaluation.
        """
        return self._inner.prepare_trials(trial_indices)

    def prepare_selections(
        self, trial_indices: Iterable[Array]
    ) -> tuple[PreparedLoss, ...]:
        """Prepare trial-index selections in order, sharing layouts when supported.

        Args:
            trial_indices: One concrete dataset-row array per selection.

        Returns:
            One prepared loss per selection, excluding first-recall and termination scores.
        """
        return self._inner.prepare_selections(trial_indices)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: jnp.ndarray,
    ) -> Float[Array, " n_samples"]:
        return self._inner(trial_indices, base_params, free_param_names, x)


# Compatibility aliases. Do not use in new code.
MemorySearchLikelihoodFnGenerator = MemorySearchLikelihoodLoss
ExcludeFirstRecallLikelihoodFnGenerator = ExcludeFirstRecallLikelihoodLoss
ExcludeTerminationLikelihoodFnGenerator = ExcludeTerminationLikelihoodLoss
