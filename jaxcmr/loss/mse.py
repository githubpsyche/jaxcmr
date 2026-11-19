"""Fit a model to behavioral measures calculated from observed and simulated recall.

Use ``MemorySearchSpcMseLoss`` when the target is recall probability by study
position, or ``MemorySearchCatSpcMseLoss`` when the target is a separate
curve for each category. To fit another measure, give ``MemorySearchMseLoss``
the analysis you use on observed recall. It applies that function to the
observed dataset and to a simulated dataset with the same study lists.

Each candidate parameter set generates several recalls per trial. The model
can stop early, and only response slots up to the observed response count
are retained. Candidates use the same random keys. With several analyses,
all predicted summaries come from the same simulated responses.
``make_factory`` lets you give a fitter that combination of analyses, their
weights, and the scales used to compare their squared errors. Discrete
recall sampling makes this loss suitable for derivative-free search.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol

import jax.numpy as jnp
from jax import random, vmap

from jaxcmr.analyses.cat_spc import cat_spc
from jaxcmr.analyses.spc import spc
from jaxcmr.dataset import recall_items_to_positions, replace_recalls, select_trials
from jaxcmr.simulation import simulate_free_recall, simulate_study
from jaxcmr.typing import (
    Array,
    ArrayLike,
    Float,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    Parameters,
    RecallDataset,
)

__all__ = [
    "MemorySearchCatSpcMseLoss",
    "MemorySearchMseLoss",
    "MemorySearchSpcMseLoss",
    "RecallAnalysisFn",
    "make_factory",
]


class RecallAnalysisFn(Protocol):
    """Calculate a fixed-shape behavioral summary from a recall dataset.

    Observations and simulations use the same study-position recall codes.
    Simulated repetitions are ordinary trial rows with repeated metadata.
    The analysis defines aggregation and weighting, must support JAX tracing,
    and may return nonfinite values for undefined observed bins.
    """

    def __call__(self, dataset: RecallDataset, /) -> Array:
        """Return a numerical summary, with the same shape for both datasets."""
        ...


class MemorySearchMseLoss:
    """Compare observed and predicted summaries with fixed weights and scales.

    Parameters
    ----------
    model_factory_cls : type[MemorySearchModelFactory]
        Factory creating each trial's model, including its specific metadata.
    dataset : RecallDataset
        Observations with position-coded recalls and trial-aligned metadata.
    features : Array or None
        Vocabulary features passed to the factory.
    analysis_fn : RecallAnalysisFn or None, optional
        A single dataset analysis, named ``analysis`` in component scores.
        Supply this or a nonempty ``analysis_fns`` mapping.
    simulation_count : int, optional
        Simulations per trial, default 20.
    analysis_fns : Mapping[str, RecallAnalysisFn] or None, optional
        Named JAX-compatible analyses accepting a RecallDataset. Their output
        shapes must agree for observations and predictions. Simulations become
        repeated trial rows, so the analysis defines pooling and weighting.
    weights : Mapping[str, float] or None, optional
        Nonnegative analysis weights, default one. Zero disables an analysis.
    scales : Mapping[str, ArrayLike] or None, optional
        Fixed positive residual divisors, broadcastable to each summary.
    reduction : {"pooled", "per_analysis"}, optional
        Pooled divides by the weighted number of valid points. Per-analysis
        averages each analysis's MSE, then takes their weighted mean.
    seed : int, optional
        Fixed simulation seed, default zero. Keys are reused across candidates.

    Notes
    -----
    Nonfinite observed bins are excluded. Nonfinite predictions at usable bins,
    or an enabled analysis with no usable observed bins, give infinite loss.
    Scales are not re-estimated when selecting participants. Configuration must
    contain at least one positive weight; weights and scales must be finite.
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
        seed: int = 0,
    ) -> None:
        self.analysis_fns = (
            {"analysis": analysis_fn}
            if analysis_fn is not None
            else dict(analysis_fns or {})
        )
        self.weights = {
            name: float((weights or {}).get(name, 1.0)) for name in self.analysis_fns
        }
        self.scales = {
            name: jnp.array((scales or {}).get(name, 1.0)) for name in self.analysis_fns
        }
        self.reduction = reduction
        self.simulation_count = simulation_count
        self.dataset = dataset.copy()
        for name, values in dataset.items():
            self.dataset[name] = jnp.asarray(values)
        self.present_lists = self.dataset["pres_itemnos"]
        self.list_length = self.present_lists.shape[1]
        self.create_model = model_factory_cls(self.dataset, features).create_trial_model
        trial_count = self.present_lists.shape[0]
        _, per_trial_key = random.split(random.PRNGKey(seed))
        self.simulation_keys = vmap(lambda key: random.split(key, simulation_count))(
            random.split(per_trial_key, trial_count)
        )
        totals = jnp.count_nonzero(self.dataset["recalls"], axis=1)
        self.trial_masks = jnp.arange(self.list_length)[None, :] < totals[:, None]

    def init_model_for_retrieval(
        self, trial_index: Integer[Array, ""], parameters: Parameters
    ) -> MemorySearch:
        """Create the trial's model and present its study list before retrieval."""
        model = self.create_model(trial_index, parameters)
        return simulate_study(model, self.present_lists[trial_index]).start_retrieving()

    def _target_analyses(
        self, trial_indices: Integer[Array, " trials"]
    ) -> dict[str, Array]:
        observed = select_trials(self.dataset, trial_indices)
        return {
            name: jnp.asarray(analysis(observed))
            for name, analysis in self.analysis_fns.items()
            if self.weights[name] > 0
        }

    def _simulate_trials(
        self, trial_indices: Integer[Array, " trials"], parameters: Parameters
    ) -> RecallDataset:
        """Simulate the selected trials, with repetition-matched metadata."""

        def simulate(
            trial_index: Integer[Array, ""],
        ) -> Integer[Array, " simulations recall_events"]:
            model = self.init_model_for_retrieval(trial_index, parameters)
            recalls = vmap(
                lambda key: simulate_free_recall(model, self.list_length, key)[1]
            )(self.simulation_keys[trial_index])
            return recalls * self.trial_masks[trial_index]

        recalls = vmap(simulate)(trial_indices).reshape(-1, self.list_length)
        repeated = select_trials(
            self.dataset, jnp.repeat(trial_indices, self.simulation_count)
        )
        positions = recall_items_to_positions(recalls, repeated["pres_itemnos"])
        return replace_recalls(repeated, positions)

    def _score_analyses(
        self, simulated: RecallDataset, targets: Mapping[str, Array]
    ) -> dict[str, Array]:
        contributions: dict[str, Array] = {}
        denominator = jnp.asarray(0.0)
        for name, analysis in self.analysis_fns.items():
            weight = self.weights[name]
            if weight == 0:
                contributions[name] = jnp.asarray(0.0)
                continue
            target, prediction = targets[name], jnp.asarray(analysis(simulated))
            valid = jnp.isfinite(target)
            count = jnp.sum(valid)
            residual = (
                jnp.where(valid, prediction, 0) - jnp.where(valid, target, 0)
            ) / self.scales[name]
            squared = jnp.square(residual)
            error = jnp.sum(jnp.where(jnp.isfinite(squared), squared, jnp.inf))
            error = jnp.where(count > 0, error, jnp.inf)
            if self.reduction == "per_analysis":
                error = error / jnp.maximum(count, 1)
                denominator = denominator + weight
            else:
                denominator = denominator + weight * count
            contributions[name] = weight * error
        denominator = jnp.where(denominator > 0, denominator, 1.0)
        return {name: value / denominator for name, value in contributions.items()}

    def present_and_predict_analysis_mse(
        self, trial_indices: Integer[Array, " trials"], parameters: Parameters
    ) -> Float[Array, ""]:
        """Return the combined error for one physical parameter mapping."""
        scores = self._score_analyses(
            self._simulate_trials(trial_indices, parameters),
            self._target_analyses(trial_indices),
        )
        return jnp.sum(jnp.stack(tuple(scores.values())))

    def component_scores(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Parameters,
        free_param_names: Iterable[str],
        x: Float[Array, " parameters samples"],
    ) -> dict[str, Float[Array, " samples"]]:
        """Return each analysis's weighted contribution for candidate columns.

        Contributions add to the total loss. Calling this separately repeats
        simulation with the same keys; it does not advance a random stream.
        """
        names = tuple(free_param_names)
        targets = self._target_analyses(trial_indices)

        def score(values: Float[Array, " parameters"]) -> dict[str, Array]:
            parameters = dict(base_params) | dict(zip(names, values))
            return self._score_analyses(
                self._simulate_trials(trial_indices, parameters), targets
            )

        return vmap(score, in_axes=1)(x)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Parameters,
        free_param_names: Iterable[str],
        x: Float[Array, " parameters samples"],
    ) -> Float[Array, " samples"]:
        """Score candidate columns using weighted behavioral squared error.
        
        Parameters
        ----------
        trial_indices : Integer[Array, "trials"]
            Dataset rows to score, preserving their order and multiplicity.
        base_params : Parameters
            Fixed model values and defaults overridden by candidate values.
        free_param_names : Iterable[str]
            Names in the order of the candidate matrix's parameter rows.
        x : Float[Array, "parameters samples"]
            Candidate columns in physical model units.
        
        Returns
        -------
        Float[Array, "samples"]
            One objective value per candidate column, in input order.
        """
        return jnp.sum(
            jnp.stack(
                tuple(
                    self.component_scores(
                        trial_indices, base_params, free_param_names, x
                    ).values()
                )
            ),
            axis=0,
        )


def make_factory(
    analysis_fns: Mapping[str, RecallAnalysisFn],
    *,
    simulation_count: int = 20,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, ArrayLike] | None = None,
    reduction: Literal["pooled", "per_analysis"] = "pooled",
    seed: int = 0,
) -> type[MemorySearchMseLoss]:
    """Bind dataset analyses and scoring settings into a fitter's loss class.

    Parameters
    ----------
    analysis_fns : Mapping[str, RecallAnalysisFn]
        Named analyses accepting RecallDataset, such as ``{"spc": spc}``.
    simulation_count : int, optional
        Repetitions per trial, default 20.
    weights : Mapping[str, float] or None, optional
        Analysis weights; see MemorySearchMseLoss.
    scales : Mapping[str, ArrayLike] or None, optional
        Fixed residual divisors; configuration arrays are copied.
    reduction : {"pooled", "per_analysis"}, optional
        Weight valid points or whole analyses, respectively.
    seed : int, optional
        Seed for fixed simulation keys, default zero.

    Returns
    -------
    type[MemorySearchMseLoss]
        Class accepting a model factory, dataset and features. Use it as the
        fitter's ``loss_fn_cls``. Configuration mappings are copied; analyses
        themselves must be pure.
    """
    analyses = dict(analysis_fns)
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
                analysis_fns=analyses,
                weights=bound_weights,
                scales=bound_scales,
                reduction=reduction,
                seed=seed,
            )

    return ConfiguredMseLoss


class MemorySearchSpcMseLoss(MemorySearchMseLoss):
    """Fit serial-position recall probabilities using 20 simulations per trial.

    ``simulation_count`` and ``seed`` are optional keyword arguments. All
    counting, padding and repeated-presentation rules come from ``spc``.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        *,
        simulation_count: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__(
            model_factory_cls, dataset, features, spc, simulation_count, seed=seed
        )


class MemorySearchCatSpcMseLoss(MemorySearchMseLoss):
    """Fit stacked category SPCs, excluding undefined category-position bins.

    Defaults are the ``condition`` field, categories (1, 2), ten simulations
    per trial and seed zero. Category summaries share one scoring denominator;
    a wholly absent category is ignored if other categories have observations.
    """

    def __init__(
        self,
        model_factory_cls: type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Float[Array, " word_pool_items features_count"] | None,
        *,
        category_field: str = "condition",
        category_values: Sequence[int] = (1, 2),
        simulation_count: int = 10,
        seed: int = 0,
    ) -> None:
        categories = tuple(category_values)

        def summary(data: RecallDataset) -> Array:
            return jnp.stack(
                [cat_spc(data, category_field, category) for category in categories]
            )

        super().__init__(
            model_factory_cls, dataset, features, summary, simulation_count, seed=seed
        )
