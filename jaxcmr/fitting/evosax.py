"""Differential-evolution fitting with ordinary or prepared jaxcmr losses.

The public workflow is fit_subjects -> fit -> _run_once -> _run_generations.
fit_once enters at _run_once with an explicit random key. LossPreparation binds
the selected observations to one loss interface before the compiled DE loop;
the loop itself only samples, evaluates and evolves parameter populations.

EvosaxDE visits subjects from Python and supports prepared losses. ScanEvosaxDE
and MapEvosaxDE execute equal-length subject selections through JAX scan/map,
using the ordinary trial-index loss interface. All three share the DE loop,
random-key convention and result format.
"""

import time
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, NamedTuple

import jax
import numpy as np
from evosax.algorithms import DifferentialEvolution
from jax import numpy as jnp
from tqdm import trange

from jaxcmr.fitting._preparation import LossPreparation
from jaxcmr.helpers import make_subject_trial_masks
from jaxcmr.loss.prepared import PreparedLoss
from jaxcmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    Float_,
    LossFn,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallDataset,
)

__all__ = ["BatchEvosaxDE", "EvosaxDE", "MapEvosaxDE", "ScanEvosaxDE"]

# Compiled runs return fitness, parameters, generation count and convergence.
type _RunArrays = tuple[Array, Array, Array, Array]


class _RestartResult(NamedTuple):
    """One completed DE restart, transferred to host values for selection/reporting."""

    fitness: float
    parameters: np.ndarray
    generations: int
    converged: bool


def _de_hyperparameters(
    options: Mapping[str, Any], parameter_count: int
) -> dict[str, Any]:
    """Normalize DE options, retaining the public population-multiplier convention."""
    pop_size = options.get("pop_size", 15)
    diff_w = options.get("diff_w", (0.5, 1.0))
    if not isinstance(diff_w, (tuple, list)):
        diff_w = (diff_w, diff_w)
    return {
        "bounds": options.get("bounds", {}),
        "num_steps": options.get("num_steps", 1000),
        "pop_size": pop_size,
        "population_size": max(4, int(pop_size) * parameter_count),
        "relative_tolerance": options.get("relative_tolerance", 0.001),
        "absolute_tolerance": options.get("absolute_tolerance", 0.0),
        "cross_over_rate": options.get("cross_over_rate", 0.9),
        "diff_w": tuple(diff_w),
        "best_of": options.get("best_of", 1),
        "init": options.get("init", "random"),
        "boundary_handling": options.get("boundary_handling", "resample"),
        "seed": int(options.get("seed", 0)),
        "nonfinite_penalty": float(options.get("nonfinite_penalty", 1e30)),
    }


class EvosaxDE:
    """Fit bounded parameters with DE restarts and optional loss preparation.

    Public fitting methods appear first, followed by selection/preparation,
    the compiled DE calculation, and result construction. Each restart starts
    a fresh population; cached selections retain data and executable code only.
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Mapping[str, Float_],
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: type[LossFn],
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure the loss, parameter bounds and reusable compiled runners.

        Args:
            dataset: Recall dataset, including one subject label per trial.
            features: Optional vocabulary feature matrix passed to the loss.
            base_params: Model defaults and fixed values; free values override them.
            model_factory_cls: Model factory passed to the loss constructor.
            loss_fn_cls: Loss class using the population-evaluation interface.
            hyperparams: Fitting options, including bounds in parameter order.
                pop_size is a multiplier of the number of free parameters;
                best_of controls independent restarts and num_steps caps generations.
        """
        options = {} if hyperparams is None else hyperparams
        self.dataset = dataset
        self.base_params = base_params
        self.subjects = dataset["subject"].flatten()
        self.progress_bar = options.get("progress_bar", True)
        self.display_iterations = options.get("display_iterations", False)
        self.free_parameter_bounds = options.get("bounds", {})
        self.bounds = np.array(list(self.free_parameter_bounds.values()))
        self.lower_bounds = jnp.asarray(self.bounds[:, 0])
        self.upper_bounds = jnp.asarray(self.bounds[:, 1])
        self.num_parameters = len(self.free_parameter_bounds)
        self.all_hyperparams = _de_hyperparameters(options, self.num_parameters)
        self._base_key = jax.random.PRNGKey(self.all_hyperparams["seed"])
        self._fit_counter = 0
        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)
        self._configure_evolution()

        self._losses = LossPreparation(self.loss_fn)
        # One stable function accepts ordinary or prepared observations as data.
        self._compiled_run = jax.jit(self._run_generations)

    def fit_once(
        self,
        trial_mask: Bool[Array, " trials"],
        key: PRNGKeyArray,
        subject_id: int = -1,
    ) -> FitResult:
        """Fit one explicit-key restart without advancing the fit-call key sequence.

        Args:
            trial_mask: Boolean mask selecting dataset rows for this fit.
            key: Random key for this restart. Independent callers can reproduce
                the restarts of a best_of fit by supplying its split keys.
            subject_id: Subject label stored with the fitted parameters.

        Returns:
            One fitted parameter vector with loss, generations, convergence and
            elapsed time. Reported best_of is one; fitter configuration is retained.
        """
        started = time.perf_counter()
        trial_indices = jnp.where(trial_mask)[0]
        prepared = self._losses.select(trial_indices)
        result = self._run_once(key, prepared)
        return self._format_result(result, subject_id, started, best_of=1)

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit selected trials and retain the lowest-loss independent restart.

        Args:
            trial_mask: Boolean mask selecting dataset rows for this fit.
            subject_id: Subject label stored with the fitted parameters.

        Returns:
            Best fitted parameters and their loss, generations and convergence.
            Elapsed time includes preparation, compilation and every restart.
        """
        started = time.perf_counter()
        trial_indices = jnp.where(trial_mask)[0]
        fit_key = self._next_key()
        prepared = self._losses.select(trial_indices)
        result = self._best_restart(fit_key, prepared)
        return self._format_result(result, subject_id, started)

    def fit_subjects(self, trial_mask: Bool[Array, " trials"]) -> FitResult:
        """Fit selected subjects independently using the serial prepared-loss path.

        Args:
            trial_mask: Boolean mask selecting trials to partition by subject.

        Returns:
            Combined results in subject order, with one value per fitted parameter
            and subject. Elapsed time covers the complete subject collection.
        """
        started = time.perf_counter()
        masks, subjects = make_subject_trial_masks(trial_mask, self.subjects)
        combined = self._empty_result()
        progress = trange(len(subjects), disable=not self.progress_bar)
        for index in progress:
            if np.sum(masks[index]) == 0:
                continue
            result = self.fit(masks[index], int(subjects[index]))
            self._append_subject_result(combined, result)
            if self.progress_bar:
                progress.set_description(
                    f"Subject={subjects[index]}, Fitness={result['fitness'][0]}"
                )
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def prepare_selections(
        self, trial_masks: Iterable[Bool[Array, " trials"]]
    ) -> float:
        """Prepare compatible layouts ahead of serial fit or fit_once calls.

        Args:
            trial_masks: Trial masks for anticipated fits. Requires the loss's
                optional batch-preparation capability. An empty collection clears
                the plan; a new collection replaces it after successful preparation.

        Returns:
            Preparation time in seconds. This call does not compile or fit;
            subsequent fit times include any required compilation. Unplanned
            selections are prepared individually. Scan/map subject methods use
            their ordinary trial-index path and do not consume this plan.
        """
        started = time.perf_counter()
        self._losses.prepare_selections(trial_masks, self.subjects.shape)
        return time.perf_counter() - started

    def _next_key(self) -> PRNGKeyArray:
        """Advance the reproducible random-key sequence by one public fit call."""
        key = jax.random.fold_in(self._base_key, self._fit_counter)
        self._fit_counter += 1
        return key

    def _best_restart(
        self, fit_key: PRNGKeyArray, prepared: PreparedLoss[Any]
    ) -> _RestartResult:
        """Run best_of fresh populations, keeping the first winner when losses tie."""
        run_keys = jax.random.split(fit_key, self.all_hyperparams["best_of"])
        best = None
        best_fitness = np.inf
        for key in run_keys:
            result = self._run_once(key, prepared)
            if result.fitness < best_fitness:
                best, best_fitness = result, result.fitness
        assert best is not None, "No fit result found"
        return best

    def _run_once(
        self, key: PRNGKeyArray, prepared: PreparedLoss[Any]
    ) -> _RestartResult:
        """Execute one restart and transfer its result to host values."""
        fitness, parameters, generations, converged = self._compiled_run(key, prepared)
        return _RestartResult(
            float(fitness),
            np.asarray(parameters, dtype=float),
            int(generations),
            bool(converged),
        )

    def _evaluate_population(
        self,
        prepared: PreparedLoss[Any],
        population: Float[Array, "candidates parameters"],
    ) -> Float[Array, " candidates"]:
        """Score parameter rows against the observations retained by a prepared loss."""
        values = prepared(self.base_params, self.free_parameter_bounds, population.T)
        return self._sanitize_fitness(values, population.shape[0])

    def _sanitize_fitness(self, values: Array, candidate_count: int) -> Array:
        """Flatten population losses and replace failed evaluations with a penalty.

        _has_converged excludes that penalty, so a population of failed candidates
        cannot be mistaken for a converged fit.
        """
        fitness = jnp.reshape(jnp.asarray(values), (candidate_count,))
        return jnp.where(
            jnp.isfinite(fitness),
            fitness,
            jnp.asarray(self.all_hyperparams["nonfinite_penalty"]),
        )

    def _configure_evolution(self) -> None:
        """Construct upstream evosax DE and configure crossover and mutation weights."""
        self._algorithm = DifferentialEvolution(
            population_size=self.all_hyperparams["population_size"],
            solution=jnp.zeros(self.num_parameters, dtype=self.lower_bounds.dtype),
        )
        self._algorithm_params = self._algorithm.default_params.replace(
            crossover_rate=self.all_hyperparams["cross_over_rate"],
            differential_weight=self.all_hyperparams["diff_w"][0],
            differential_weight_min=self.all_hyperparams["diff_w"][0],
            differential_weight_max=self.all_hyperparams["diff_w"][1],
        )

    def _run_generations(
        self, key: PRNGKeyArray, prepared: PreparedLoss[Any]
    ) -> _RunArrays:
        """Initialize and evolve one population against its selected observations.

        PreparedLoss supplies one scoring interface and dynamic data for every
        loss route. This loop has no selection caches or preparation branches.
        Returns fitness, parameters, generations and convergence as JAX arrays.
        """
        evaluate = partial(self._evaluate_population, prepared)
        init_key, tell_key, loop_key = jax.random.split(key, 3)
        population = self._sample_initial_population(init_key)
        fitness = evaluate(population)
        state = self._algorithm.init(
            init_key, population, fitness, self._algorithm_params
        )
        state, _ = self._algorithm.tell(
            tell_key, population, fitness, state, self._algorithm_params
        )

        def evolve(carry):
            loop_key, state, _, generations = carry
            loop_key, ask_key, bounds_key, tell_key = jax.random.split(loop_key, 4)
            population, state = self._algorithm.ask(
                ask_key, state, self._algorithm_params
            )
            population = self._ensure_bounds(bounds_key, population)
            fitness = evaluate(population)
            state, _ = self._algorithm.tell(
                tell_key, population, fitness, state, self._algorithm_params
            )
            return loop_key, state, self._has_converged(state.fitness), generations + 1

        def step(carry, _):
            # Keep the fixed scan shape but skip evolution after convergence.
            return jax.lax.cond(carry[2], lambda current: current, evolve, carry), None

        (_, state, converged, generations), _ = jax.lax.scan(
            step,
            (loop_key, state, jnp.asarray(False), jnp.asarray(0)),
            xs=None,
            length=int(self.all_hyperparams["num_steps"]),
        )
        return (
            state.best_fitness,
            self._algorithm.get_best_solution(state),
            generations,
            converged,
        )

    def _sample_initial_population(self, key: PRNGKeyArray) -> Array:
        """Sample bounded parameter rows, uniformly or by Latin hypercube.

        Latin-hypercube initialization stratifies each parameter's unit interval
        and independently permutes its strata before converting to physical bounds.
        """
        if self.all_hyperparams["init"] == "latinhypercube":
            sample_key, permutation_key = jax.random.split(key)
            segment_size = 1.0 / self.all_hyperparams["population_size"]
            samples = (
                segment_size
                * jax.random.uniform(
                    sample_key,
                    (self.all_hyperparams["population_size"], self.num_parameters),
                )
                + jnp.linspace(
                    0.0, 1.0, self.all_hyperparams["population_size"], endpoint=False
                )[:, None]
            )
            permutation_keys = jax.random.split(permutation_key, self.num_parameters)
            unit_population = jax.vmap(
                lambda values, key: jax.random.permutation(key, values),
                in_axes=(1, 0),
                out_axes=1,
            )(samples, permutation_keys)
        else:
            unit_population = jax.random.uniform(
                key, (self.all_hyperparams["population_size"], self.num_parameters)
            )
        return self.lower_bounds + unit_population * (
            self.upper_bounds - self.lower_bounds
        )

    def _ensure_bounds(self, key: PRNGKeyArray, population: Array) -> Array:
        """Repair individual out-of-bounds coordinates by resampling or clipping."""
        out_of_bounds = jnp.logical_or(
            population < self.lower_bounds, population > self.upper_bounds
        )
        if self.all_hyperparams["boundary_handling"] == "resample":
            replacement = self.lower_bounds + jax.random.uniform(
                key, population.shape, dtype=population.dtype
            ) * (self.upper_bounds - self.lower_bounds)
            return jnp.where(out_of_bounds, replacement, population)
        return jnp.clip(population, self.lower_bounds, self.upper_bounds)

    def _has_converged(self, fitness: Array) -> Array:
        """Check SciPy-style population-loss spread, excluding penalized candidates."""
        penalty = jnp.asarray(
            self.all_hyperparams["nonfinite_penalty"], dtype=fitness.dtype
        )
        threshold = self.all_hyperparams["absolute_tolerance"] + self.all_hyperparams[
            "relative_tolerance"
        ] * jnp.abs(jnp.mean(fitness))
        return (
            jnp.all(fitness != penalty)
            & jnp.all(jnp.isfinite(fitness))
            & (jnp.std(fitness) <= threshold)
        )

    def _empty_result(self) -> FitResult:
        """Create the shared result schema before adding any fitted subjects."""
        return {
            "fixed": {name: float(value) for name, value in self.base_params.items()},
            "free": dict(self.free_parameter_bounds),
            "fitness": [],
            "nit": [],
            "converged": [],
            "fits": {
                **{name: [] for name in self.base_params},
                **{name: [] for name in self.free_parameter_bounds},
                "subject": [],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": 0.0,
        }

    def _format_result(
        self,
        outcome: _RestartResult,
        subject_id: int,
        started: float,
        *,
        best_of: int | None = None,
    ) -> FitResult:
        """Package one restart winner, letting fitted values override base defaults."""
        result = self._empty_result()
        result["fitness"] = [outcome.fitness]
        result["nit"] = [outcome.generations]
        result["converged"] = [outcome.converged]
        result["fits"] = {
            **{name: [float(value)] for name, value in self.base_params.items()},
            **{
                name: [float(outcome.parameters[index])]
                for index, name in enumerate(self.free_parameter_bounds)
            },
            "subject": [subject_id],
        }
        if best_of is not None:
            result["hyperparameters"] = dict(self.all_hyperparams, best_of=best_of)
        result["fit_time"] = time.perf_counter() - started
        return result

    @staticmethod
    def _append_subject_result(combined: FitResult, result: FitResult) -> None:
        """Append one subject's values to a cohort result and reject nonfinite loss."""
        combined["fitness"].extend(result["fitness"])
        combined["nit"].extend(result["nit"])
        combined["converged"].extend(result["converged"])
        for name, values in result["fits"].items():
            combined["fits"][name].extend(values)
        if not np.isfinite(result["fitness"][0]):
            raise ValueError(
                f"Non-finite fitness for subject {result['fits']['subject'][0]}", result
            )


class ScanEvosaxDE(EvosaxDE):
    """Fit equal-length subject selections with an outer compiled JAX scan.

    Subject fitting uses ordinary trial-index losses. Inherited fit and fit_once
    retain the serial class's optional preparation support.
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Mapping[str, Float_],
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: type[LossFn],
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure DE as in EvosaxDE, then create a compiled subject runner."""
        super().__init__(
            dataset,
            features,
            base_params,
            model_factory_cls,
            loss_fn_cls,
            hyperparams=hyperparams,
        )
        self._compiled_subject_runs = jax.jit(self._run_subjects)

    def fit_subjects(self, trial_mask: Bool[Array, " trials"]) -> FitResult:
        """Fit each subject, retaining its best restart across compiled subject runs.

        Args:
            trial_mask: Boolean mask selecting trials to partition by subject.
                Nonempty selections must have the same trial count.

        Returns:
            Combined results in subject order. Random keys match serial subject
            fitting, and total time includes compilation and all restarts.
        """
        started = time.perf_counter()
        subjects, trial_indices = self._subject_selections(trial_mask)
        combined = self._empty_result()
        if len(subjects):
            outcomes = self._best_subject_restarts(trial_indices)
            for subject, outcome in zip(subjects, outcomes, strict=True):
                result = self._format_result(outcome, int(subject), started)
                self._append_subject_result(combined, result)
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def _subject_selections(self, trial_mask: Array) -> tuple[np.ndarray, Array]:
        """Select nonempty subjects and construct a rectangular trial-index matrix."""
        masks, subjects = make_subject_trial_masks(trial_mask, self.subjects)
        valid = [index for index in range(len(subjects)) if np.sum(masks[index]) != 0]
        if not valid:
            return np.asarray(subjects)[valid], jnp.empty((0, 0), dtype=jnp.int32)
        indices = [np.where(masks[index])[0] for index in valid]
        if len({len(selected) for selected in indices}) != 1:
            raise ValueError(
                "ScanEvosaxDE.fit_subjects requires the same number of selected trials per subject."
            )
        return np.asarray(subjects)[valid], jnp.asarray(indices)

    def _subject_restart_keys(self, subject_count: int) -> Array:
        """Reserve the same per-fit random streams as serial subject fitting."""
        fit_keys = jnp.asarray(
            [
                jax.random.fold_in(self._base_key, self._fit_counter + index)
                for index in range(subject_count)
            ]
        )
        self._fit_counter += subject_count
        return jax.vmap(
            lambda key: jax.random.split(key, self.all_hyperparams["best_of"])
        )(fit_keys)

    def _best_subject_restarts(self, trial_indices: Array) -> list[_RestartResult]:
        """Run each restart across subjects and retain winners independently."""
        keys = self._subject_restart_keys(len(trial_indices))
        initial = self._compiled_subject_runs(keys[:, 0, :], trial_indices)
        best_fitness, best_parameters, best_generations, best_converged = (
            np.asarray(initial[0], dtype=float),
            np.asarray(initial[1], dtype=float),
            np.asarray(initial[2], dtype=int),
            np.asarray(initial[3], dtype=bool),
        )
        for restart in trange(
            1,
            self.all_hyperparams["best_of"],
            disable=not self.progress_bar or self.all_hyperparams["best_of"] <= 1,
        ):
            raw = self._compiled_subject_runs(keys[:, restart, :], trial_indices)
            fitness, parameters, generations, converged = (
                np.asarray(raw[0], dtype=float),
                np.asarray(raw[1], dtype=float),
                np.asarray(raw[2], dtype=int),
                np.asarray(raw[3], dtype=bool),
            )
            improved = fitness < best_fitness
            best_fitness = np.where(improved, fitness, best_fitness)
            best_parameters = np.where(improved[:, None], parameters, best_parameters)
            best_generations = np.where(improved, generations, best_generations)
            best_converged = np.where(improved, converged, best_converged)
        return [
            _RestartResult(
                float(fitness), parameters, int(generations), bool(converged)
            )
            for fitness, parameters, generations, converged in zip(
                best_fitness,
                best_parameters,
                best_generations,
                best_converged,
                strict=True,
            )
        ]

    def _run_subjects(
        self, subject_keys: Array, subject_trial_indices: Array
    ) -> _RunArrays:
        """Scan ordinary-loss DE runs over equally sized trial-index selections."""

        def step(carry, selection):
            key, indices = selection
            return carry, self._run_generations(key, self._losses.ordinary(indices))

        _, outputs = jax.lax.scan(step, None, (subject_keys, subject_trial_indices))
        return outputs


class MapEvosaxDE(ScanEvosaxDE):
    """Use JAX map for equal-length subject fits, with optional subject batching."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Mapping[str, Float_],
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: type[LossFn],
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure DE, optionally setting map_batch_size in the hyperparameters.

        None maps subjects sequentially, a positive size batches subjects, and
        zero requests JAX's full-vmap path. Other options match EvosaxDE.
        """
        if hyperparams is None:
            hyperparams = {}
        self.map_batch_size = hyperparams.get("map_batch_size")
        if self.map_batch_size is not None:
            self.map_batch_size = int(self.map_batch_size)
        if self.map_batch_size is not None and self.map_batch_size < 0:
            raise ValueError(
                "MapEvosaxDE requires map_batch_size to be None or a non-negative integer."
            )
        super().__init__(
            dataset,
            features,
            base_params,
            model_factory_cls,
            loss_fn_cls,
            hyperparams=hyperparams,
        )
        self.all_hyperparams["map_batch_size"] = self.map_batch_size

    def _run_subjects(
        self, subject_keys: Array, subject_trial_indices: Array
    ) -> _RunArrays:
        """Map ordinary-loss DE runs using the configured subject batch size."""

        def fit_subject(selection):
            key, indices = selection
            return self._run_generations(key, self._losses.ordinary(indices))

        return jax.lax.map(
            fit_subject,
            (subject_keys, subject_trial_indices),
            batch_size=self.map_batch_size,
        )


# Compatibility alias retained for existing artifacts.
BatchEvosaxDE = ScanEvosaxDE
