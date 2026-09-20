"""Fit one or more recall summaries with a configurable squared-error loss.

``make_factory`` binds analyses and scoring options into a loss class accepted
by the existing fitters. All analyses share simulations for each candidate.
"""

from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np
from jax import lax, random, vmap
from jax import numpy as jnp

from jaxcmr.simulation import simulate_free_recall
from jaxcmr.typing import (
    Array,
    ArrayLike,
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
    "MemorySearchMseLoss",
    "make_factory",
    "simulate_masked_free_recall",
]


def simulate_masked_free_recall(
    model: MemorySearch,
    list_length: int,
    mask: jnp.ndarray,
    keys: PRNGKeyArray,
) -> Integer[Array, "simulation_count recall_length"]:
    """Simulate recall sequences and zero slots beyond the observed count."""

    def simulate_once(key: PRNGKeyArray) -> Integer[Array, " recall_length"]:
        _, recalls = simulate_free_recall(model, list_length, key)
        return recalls * mask

    return vmap(simulate_once)(keys)


class MemorySearchMseLoss:
    """Compare observed and simulated summaries using fixed scoring choices.

    Prefer ``make_factory`` when passing this loss to a fitter. Direct
    construction still accepts a single ``analysis_fn`` for compatibility.

    Analyses must be JAX-compatible and return matching observed/simulated
    shapes. Observed recalls have one simulation per trial; simulated recalls
    have ``simulation_count``. Callbacks control aggregation, bin selection,
    and contrasts. The full dataset and selected trial indices accompany both.

    This loss retains the existing simulation convention: model stopping is
    active, then responses beyond each trial's observed recall count are
    masked. It does not convert recall codes: callbacks must receive comparable
    observed and model item codes. Simulations have study-list width, which may
    differ from the observed recall width. Discrete sampling does not supply
    useful pathwise gradients merely because the loss is JAX-compilable.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        analysis_fn: RecallAnalysisFn | None = None,
        simulation_count: int = 20,
        *,
        analysis_fns: Mapping[str, RecallAnalysisFn] | None = None,
        weights: Mapping[str, float] | None = None,
        scales: Mapping[str, ArrayLike] | None = None,
        reduction: Literal["pooled", "per_analysis"] = "pooled",
    ) -> None:
        """Bind dataset/model inputs; see ``make_factory`` for scoring options.

        Supply either ``analysis_fn`` (named ``"analysis"`` in diagnostics)
        or a nonempty mapping of named ``analysis_fns``.
        """
        if (analysis_fn is None) == (analysis_fns is None):
            raise ValueError("Supply exactly one of analysis_fn or analysis_fns.")
        if analysis_fn is not None:
            analysis_fns = {"analysis": analysis_fn}
        assert analysis_fns is not None
        if not analysis_fns:
            raise ValueError("analysis_fns must contain at least one analysis.")
        if reduction not in ("pooled", "per_analysis"):
            raise ValueError("reduction must be 'pooled' or 'per_analysis'.")
        if (
            isinstance(simulation_count, bool)
            or not isinstance(simulation_count, int)
            or simulation_count < 1
        ):
            raise ValueError("simulation_count must be a positive integer.")

        self.analysis_fns = dict(analysis_fns)
        for option, values in (("weights", weights), ("scales", scales)):
            if values is not None:
                unknown = values.keys() - self.analysis_fns.keys()
                if unknown:
                    raise ValueError(
                        f"Unknown analysis names in {option}: {sorted(unknown)}"
                    )
        self.weights = {
            name: float(weights[name]) if weights is not None and name in weights else 1.0
            for name in self.analysis_fns
        }
        if any(not np.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("Analysis weights must be finite and nonnegative.")
        if not any(self.weights.values()):
            raise ValueError("At least one analysis must have positive weight.")
        self.scales = {
            name: (
                jnp.array(scales[name])
                if scales is not None and name in scales
                else jnp.asarray(1.0)
            )
            for name in self.analysis_fns
        }
        for name, scale in self.scales.items():
            if not np.all(np.isfinite(scale) & (np.asarray(scale) > 0)):
                raise ValueError(f"Scale for {name!r} must be finite and positive.")
        self.reduction = reduction
        self.simulation_count = simulation_count

        factory = model_factory_cls(dataset, features)
        self.create_model = factory.create_trial_model
        self.dataset_view = {key: jnp.asarray(value) for key, value in dataset.items()}
        self.present_lists = jnp.asarray(self.dataset_view["pres_itemnos"], dtype=jnp.int32)
        self.dataset_view["pres_itemnos"] = self.present_lists
        self.list_length = int(self.present_lists.shape[1])
        self.dataset_recalls = jnp.asarray(self.dataset_view["recalls"], dtype=jnp.int32)
        self.dataset_view["recalls"] = self.dataset_recalls

        # Fixed random numbers across candidates reduce Monte Carlo differences.
        trial_count = int(self.dataset_recalls.shape[0])
        shared_key, per_trial_key = random.split(random.PRNGKey(0))
        self.shared_simulation_keys = random.split(shared_key, simulation_count)
        self.simulation_keys = vmap(lambda key: random.split(key, simulation_count))(
            random.split(per_trial_key, trial_count)
        )
        recall_totals = jnp.count_nonzero(self.dataset_recalls, axis=1)
        self.trial_masks = jnp.arange(self.list_length)[None, :] < recall_totals[:, None]

    def init_model_for_retrieval(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Return the model prepared for retrieval on the specified trial."""
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
        return model.start_retrieving()

    def _target_analyses(
        self, trial_indices: Integer[Array, " trials"]
    ) -> dict[str, Array]:
        """Summarize observed data independently of candidate parameters."""
        observed = self.dataset_recalls[trial_indices][:, None, :]
        return {
            name: jnp.asarray(
                fn(
                    observed,
                    list_length=self.list_length,
                    dataset=self.dataset_view,
                    trial_indices=trial_indices,
                )
            )
            for name, fn in self.analysis_fns.items()
            if self.weights[name] > 0
        }

    def _score_analyses(
        self,
        simulated: Integer[Array, " trials simulations recall_positions"],
        trial_indices: Integer[Array, " trials"],
        targets: Mapping[str, Array],
    ) -> dict[str, Array]:
        """Return additive contributions, with masks based only on observations."""
        contributions: dict[str, Array] = {}
        denominator = jnp.asarray(0.0)
        for name, fn in self.analysis_fns.items():
            weight = self.weights[name]
            if weight == 0:
                contributions[name] = jnp.asarray(0.0)
                continue
            target = targets[name]
            prediction = jnp.asarray(
                fn(
                    simulated,
                    list_length=self.list_length,
                    dataset=self.dataset_view,
                    trial_indices=trial_indices,
                )
            )
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Analysis {name!r} returned observed shape {target.shape} "
                    f"and simulated shape {prediction.shape}."
                )
            try:
                scale = jnp.broadcast_to(self.scales[name], target.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Scale for {name!r} cannot broadcast to analysis shape {target.shape}."
                ) from exc

            valid = jnp.isfinite(target)
            count = jnp.sum(valid)
            residual = (
                jnp.where(valid, prediction, 0) - jnp.where(valid, target, 0)
            ) / scale
            squared = jnp.square(residual)
            error = jnp.sum(jnp.where(jnp.isfinite(squared), squared, jnp.inf))
            # Missing predictions never remove an observed bin. Entirely missing
            # targets must not let an active analysis report a perfect fit.
            error = jnp.where(count > 0, error, jnp.inf)
            if self.reduction == "per_analysis":
                error = error / jnp.maximum(count, 1)
                denominator = denominator + weight
            else:
                denominator = denominator + weight * count
            contributions[name] = weight * error

        denominator = jnp.where(denominator > 0, denominator, 1.0)
        return {name: value / denominator for name, value in contributions.items()}

    def _simulate_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Integer[Array, " trials simulations recall_positions"]:
        """Simulate every trial's model, including its specific metadata."""
        models = vmap(self.init_model_for_retrieval, in_axes=(0, None))(
            trial_indices, parameters
        )
        return vmap(simulate_masked_free_recall, in_axes=(0, None, 0, 0))(
            models,
            self.list_length,
            self.trial_masks[trial_indices],
            self.simulation_keys[trial_indices],
        )

    def base_analysis_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Score shared simulations when trials have identical prepared models.

        This explicit shortcut requires model equivalence, including metadata.
        The fitter-facing interface always prepares models separately per trial.
        """
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        simulated = vmap(simulate_masked_free_recall, in_axes=(None, None, 0, None))(
            model,
            self.list_length,
            self.trial_masks[trial_indices],
            self.shared_simulation_keys,
        )
        scores = self._score_analyses(
            simulated, trial_indices, self._target_analyses(trial_indices)
        )
        return jnp.sum(jnp.stack(tuple(scores.values())))

    def present_and_predict_analysis_mse(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Return combined MSE for one parameter set with per-trial models."""
        scores = self._score_analyses(
            self._simulate_trials(trial_indices, parameters),
            trial_indices,
            self._target_analyses(trial_indices),
        )
        return jnp.sum(jnp.stack(tuple(scores.values())))

    def component_scores(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: Float[Array, " parameters n_samples"],
    ) -> dict[str, Float[Array, " n_samples"]]:
        """Return named weighted contributions whose sum equals the total loss.

        Arguments and candidate orientation match ``__call__``. This method is
        also JIT/vmap-compatible. Calling it separately runs simulations again
        with the same fixed random numbers.
        """
        if trial_indices.size == 0:
            raise ValueError("Select at least one trial for the loss.")
        free_param_names = tuple(free_param_names)
        targets = self._target_analyses(trial_indices)

        def score_one(x_row: Array) -> dict[str, Array]:
            parameters = {
                **base_params,
                **{name: x_row[i] for i, name in enumerate(free_param_names)},
            }
            simulated = self._simulate_trials(trial_indices, parameters)
            return self._score_analyses(simulated, trial_indices, targets)

        return vmap(score_one, in_axes=1)(x)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: Float[Array, " parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        """Return one combined loss per candidate parameter vector."""
        scores = self.component_scores(trial_indices, base_params, free_param_names, x)
        return jnp.sum(jnp.stack(tuple(scores.values())), axis=0)


def make_factory(
    analysis_fns: Mapping[str, RecallAnalysisFn],
    *,
    simulation_count: int = 20,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, ArrayLike] | None = None,
    reduction: Literal["pooled", "per_analysis"] = "pooled",
) -> type[MemorySearchMseLoss]:
    """Return a configured loss class accepted as a fitter's ``loss_fn_cls``.

    Args:
        analysis_fns: Named ``RecallAnalysisFn`` callbacks. Each receives a
            trial-by-simulation-by-output recall array, the full dataset, and
            selected trial indices. Dataset-based analyses need an adapter to
            align/replicate metadata and aggregate the simulation dimension.
        simulation_count: Positive number of simulations per trial, shared by
            all analyses. Random numbers remain fixed across candidates.
        weights: Optional nonnegative weights by analysis name (default 1).
            Zero disables an analysis; at least one weight must be positive.
        scales: Optional positive finite residual divisors, scalar or broadcastable
            to each analysis's output (default 1). Empirical standard errors
            yield squared standardized errors. Scales are supplied externally,
            remain fixed across candidates, and must be appropriate for the
            trials being fitted; they are not re-estimated for subject subsets.
        reduction: ``"pooled"`` divides the weighted sum of squared residuals
            by the weighted count of valid points. ``"per_analysis"`` takes
            the weighted mean of each analysis's mean squared error, giving
            equally weighted analyses equal influence regardless of bin count.

    Nonfinite observed bins are excluded using an observation-only mask. A
    nonfinite prediction on a valid bin, or an active analysis with no valid
    observed bins, gives infinite loss. Output shapes must match exactly.
    Configuration mappings are copied; callbacks themselves must be pure.
    With unit weights and standard-error scales, pooled MSE equals the sum of
    squared standardized residuals divided by the retained point count.

    Example::

        BehavioralLoss = make_factory(
            {"spc": spc_summary, "pfr": pfr_summary, "lag_crp": lag_crp_summary},
            weights={"spc": 1.0, "pfr": 1.0, "lag_crp": 2.0},
            simulation_count=100,
        )
        # Pass loss_fn_cls=BehavioralLoss to ScipyDE or EvosaxDE.
        loss = BehavioralLoss(model_factory_cls, dataset, features)
        parts = loss.component_scores(trial_indices, base_params, free_names, x)
        total = loss(trial_indices, base_params, free_names, x)
    """
    bound_analyses = dict(analysis_fns)
    bound_weights = dict(weights) if weights is not None else None
    bound_scales = (
        {name: jnp.array(value) for name, value in scales.items()}
        if scales is not None
        else None
    )

    class ConfiguredMseLoss(MemorySearchMseLoss):
        def __init__(
            self,
            model_factory_cls: type[MemorySearchModelFactory],
            dataset: RecallDataset,
            features: Float[Array, " word_pool_items features_count"] | None,
        ) -> None:
            super().__init__(
                model_factory_cls,
                dataset,
                features,
                simulation_count=simulation_count,
                analysis_fns=bound_analyses,
                weights=bound_weights,
                scales=bound_scales,
                reduction=reduction,
            )

    return ConfiguredMseLoss


# Compatibility aliases. Do not use in new code.
MemorySearchMseFnGenerator = MemorySearchMseLoss
