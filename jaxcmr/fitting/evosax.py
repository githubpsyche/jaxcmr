"""Differential-evolution fitting using the ordinary trial-index loss interface.

Serial, scanned and mapped subject drivers share the numerical DE calculation,
random-key convention and result format. Loss evaluation receives the selected
trial indices directly; no observation preparation is performed here.
"""

import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import jax
import numpy as np
from evosax.algorithms import DifferentialEvolution
from evosax.algorithms.population_based.differential_evolution import State as DEState
from jax import numpy as jnp
from jaxtyping import UInt32
from tqdm import trange

from jaxcmr.dataset import make_subject_trial_masks
from jaxcmr.fitting.parameters import ParameterSpace
from jaxcmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    Integer,
    LossFactory,
    MemorySearchModelFactory,
    Parameters,
    PRNGKeyArray,
    RecallDataset,
)

__all__ = ["EvosaxDE", "MapEvosaxDE", "ScanEvosaxDE"]

# Compiled runs return fitness, parameters, generation count and convergence.
type _RunArrays = tuple[
    Float[Array, "*batch"],
    Float[Array, "*batch parameters"],
    Integer[Array, "*batch"],
    Bool[Array, "*batch"],
]


class _RestartResult(NamedTuple):
    """One completed DE restart, transferred to host values for selection/reporting."""

    fitness: float
    parameters: Float[np.ndarray, " parameters"]
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
        "transforms": dict(options.get("transforms", {})),
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
    """Fit trial-index losses with differential evolution and independent restarts.

    Each restart uses a fresh population. The serial, scanned and mapped fitters
    share the same numerical search and random-key schedule. Scanned and mapped
    participant fitting require equal selected trial counts across participants.

    The hyperparams mapping sets bounds, pop_size (a multiplier of the number of
    free parameters), best_of, num_steps and seed. Optional transforms select
    ParameterSpace search coordinates; omission preserves physical-coordinate
    search. No observation preparation is discovered or cached by these fitters.
    """

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Parameters,
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: LossFactory,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure the objective and bounded search.

        Parameters
        ----------
        dataset : RecallDataset
            Trial data, with participant identifiers used by fit_subjects.
        features : Array or None
            Vocabulary features passed to the loss constructor.
        base_params : Parameters
            Fixed model values and defaults overridden by fitted values.
        model_factory_cls : type[MemorySearchModelFactory]
            Factory used by the loss to construct trial-specific models.
        loss_fn_cls : LossFactory
            Loss class or configured callable, such as functools.partial. It accepts
            the model factory, dataset and features and returns a population scorer.
        hyperparams : dict or None, optional
            Search settings, including a required bounds mapping in physical units.
            Bounds order defines the free-parameter order. Transforms are opt-in;
            losses and returned parameters always use physical units.
        """
        self._configure_optimizer(dataset, base_params, hyperparams)
        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)
        self._compiled_run = jax.jit(self._run_selected_trials)

    def _configure_optimizer(
        self,
        dataset: RecallDataset,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None,
    ) -> None:
        """Configure evolution and result reporting independently of the scorer."""
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
        self.parameter_space = (
            ParameterSpace(self.free_parameter_bounds, options["transforms"])
            if options.get("transforms")
            else None
        )
        if self.parameter_space is not None:
            self.lower_bounds = jnp.zeros_like(self.lower_bounds)
            self.upper_bounds = jnp.ones_like(self.upper_bounds)
        self.num_parameters = len(self.free_parameter_bounds)
        self.all_hyperparams = _de_hyperparameters(options, self.num_parameters)
        self._base_key = jax.random.PRNGKey(self.all_hyperparams["seed"])
        self._fit_counter = 0
        self._configure_evolution()

    def fit_once(
        self,
        trial_mask: Bool[Array, " trials"],
        key: PRNGKeyArray,
        subject_id: int = -1,
    ) -> FitResult:
        """Fit one restart using an explicit random key.

        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Nonempty trial selection.
        key : PRNGKeyArray
            Restart key; this call does not advance the fitter's fit-call sequence.
        subject_id : int, optional
            Result label, default -1.

        Returns
        -------
        FitResult
            One fitted row and a one-entry restart history. Reported best_of is one;
            the fitter's settings are unchanged.
        """
        started = time.perf_counter()
        trial_indices = jnp.where(trial_mask)[0]
        result = self._run_once(key, trial_indices)
        return self._format_result([result], subject_id, started, best_of=1)

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit selected trials and retain the best usable restart.

        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Select at least one dataset row. Selected rows share fitted parameters.
        subject_id : int, optional
            Result label, default -1 for a pooled fit. This label does not select rows.

        Returns
        -------
        FitResult
            One row of physical parameters and objective values. Its one-entry
            diagnostics list retains all attempted restarts and their selected index.
            Elapsed time includes compilation and every restart.
        """
        started = time.perf_counter()
        trial_indices = jnp.where(trial_mask)[0]
        fit_key = self._next_key()
        result = self._run_restarts(
            fit_key, partial(self._run_once, indices=trial_indices)
        )
        return self._format_result(result, subject_id, started)

    def fit_subjects(self, trial_mask: Bool[Array, " trials"]) -> FitResult:
        """Fit participants separately in participant-ID order.

        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Trial selection, partitioned using the dataset's participant identifiers.

        Returns
        -------
        FitResult
            Parameter columns, fitness values and diagnostics in the same participant
            order. Elapsed time covers the complete collection of fits.
        """
        started = time.perf_counter()
        masks, subjects = make_subject_trial_masks(trial_mask, self.subjects)
        combined = self._empty_result()
        progress = trange(len(subjects), disable=not self.progress_bar)
        for index in progress:
            result = self.fit(masks[index], int(subjects[index]))
            self._append_subject_result(combined, result)
            if self.progress_bar:
                progress.set_description(
                    f"Subject={subjects[index]}, Fitness={result['fitness'][0]}"
                )
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def _next_key(self) -> PRNGKeyArray:
        """Advance the reproducible random-key sequence by one public fit call."""
        key = jax.random.fold_in(self._base_key, self._fit_counter)
        self._fit_counter += 1
        return key

    def _run_restarts(
        self, fit_key: PRNGKeyArray, run_once: Callable[[PRNGKeyArray], _RestartResult]
    ) -> list[_RestartResult]:
        """Run independent populations, retaining every restart for reporting."""
        run_keys = jax.random.split(fit_key, self.all_hyperparams["best_of"])
        return [run_once(key) for key in run_keys]

    def _run_once(
        self, key: PRNGKeyArray, indices: Integer[Array, " selected_trials"]
    ) -> _RestartResult:
        """Execute one restart and transfer its result to host values."""
        fitness, parameters, generations, converged = self._compiled_run(key, indices)
        return _RestartResult(
            float(fitness),
            np.asarray(parameters, dtype=float),
            int(generations),
            bool(converged),
        )

    def _evaluate_population(
        self,
        indices: Integer[Array, " selected_trials"],
        population: Float[Array, "candidates parameters"],
    ) -> Float[Array, " candidates"]:
        """Score parameter rows on the selected trials."""
        values = self.loss_fn(
            indices, self.base_params, self.free_parameter_bounds, population.T
        )
        return self._sanitize_fitness(values, population.shape[0])

    def _sanitize_fitness(
        self,
        values: Float[Array, " samples"],
        candidate_count: int,
    ) -> Float[Array, " samples"]:
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

    def _run_selected_trials(
        self,
        key: PRNGKeyArray,
        indices: Integer[Array, " selected_trials"],
    ) -> _RunArrays:
        """Bind dynamic trial indices before entering the shared evolution loop."""
        return self._run_generations(key, partial(self._evaluate_population, indices))

    def _run_generations(
        self,
        key: PRNGKeyArray,
        evaluate: Callable[
            [Float[Array, "candidates parameters"]], Float[Array, " candidates"]
        ],
    ) -> _RunArrays:
        """Initialize and evolve a population using its supplied numerical scorer."""
        if self.parameter_space is not None:
            score_physical = evaluate
            space = self.parameter_space
            evaluate = lambda population: score_physical(space.to_physical(population))
        init_key, tell_key, loop_key = jax.random.split(key, 3)
        population = self._sample_initial_population(init_key)
        fitness = evaluate(population)
        state = self._algorithm.init(
            init_key, population, fitness, self._algorithm_params
        )
        state, _ = self._algorithm.tell(
            tell_key, population, fitness, state, self._algorithm_params
        )

        def evolve(
            carry: tuple[PRNGKeyArray, DEState, Bool[Array, ""], Integer[Array, ""]],
        ) -> tuple[PRNGKeyArray, DEState, Bool[Array, ""], Integer[Array, ""]]:
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

        def step(
            carry: tuple[PRNGKeyArray, DEState, Bool[Array, ""], Integer[Array, ""]],
            _: None,
        ) -> tuple[
            tuple[PRNGKeyArray, DEState, Bool[Array, ""], Integer[Array, ""]], None
        ]:
            # Keep the fixed scan shape but skip evolution after convergence.
            return jax.lax.cond(carry[2], lambda current: current, evolve, carry), None

        (_, state, converged, generations), _ = jax.lax.scan(
            step,
            (loop_key, state, jnp.asarray(False), jnp.asarray(0)),
            xs=None,
            length=int(self.all_hyperparams["num_steps"]),
        )
        parameters = self._algorithm.get_best_solution(state)
        if self.parameter_space is not None:
            parameters = self.parameter_space.to_physical(parameters)
        return (
            jnp.asarray(state.best_fitness),
            parameters,
            generations,
            converged,
        )

    def _sample_initial_population(
        self, key: PRNGKeyArray
    ) -> Float[Array, " samples parameters"]:
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

    def _ensure_bounds(
        self,
        key: PRNGKeyArray,
        population: Float[Array, " samples parameters"],
    ) -> Float[Array, " samples parameters"]:
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

    def _has_converged(self, fitness: Float[Array, " samples"]) -> Bool[Array, ""]:
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
            "fixed": {
                name: np.asarray(value).item()
                for name, value in self.base_params.items()
            },
            "free": dict(self.free_parameter_bounds),
            "fitness": [],
            "diagnostics": [],
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
        outcomes: list[_RestartResult],
        subject_id: int,
        started: float,
        *,
        best_of: int | None = None,
    ) -> FitResult:
        """Package one restart winner, letting fitted values override base defaults."""
        restarts = []
        for index, outcome in enumerate(outcomes):
            usable = (
                np.isfinite(outcome.fitness)
                and outcome.fitness != self.all_hyperparams["nonfinite_penalty"]
            )
            restarts.append(
                {
                    "restart": index,
                    "fitness": outcome.fitness if usable else None,
                    "parameters": dict(
                        zip(
                            self.free_parameter_bounds,
                            map(float, outcome.parameters),
                            strict=True,
                        )
                    )
                    if usable
                    else None,
                    "nit": outcome.generations,
                    "converged": outcome.converged if usable else False,
                }
            )
        usable_indices = [
            i for i, record in enumerate(restarts) if record["fitness"] is not None
        ]
        if not usable_indices:
            raise ValueError(
                f"No usable evaluation was found. Restart diagnostics: {restarts}"
            )
        selected = min(usable_indices, key=lambda i: outcomes[i].fitness)
        outcome = outcomes[selected]
        result = self._empty_result()
        result["diagnostics"] = [
            {"subject": subject_id, "selected_restart": selected, "restarts": restarts}
        ]
        result["fitness"] = [outcome.fitness]
        result["nit"] = [outcome.generations]
        result["converged"] = [outcome.converged]
        result["fits"] = {
            **{
                name: [np.asarray(value).item()]
                for name, value in self.base_params.items()
            },
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
        combined.setdefault("diagnostics", []).extend(result.get("diagnostics", []))
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
        base_params: Parameters,
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: LossFactory,
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
        """Fit participants through a compiled scan.
        
        Parameters
        ----------
        trial_mask : Bool[Array, "trials"]
            Selected trials. Every included participant must have the same trial count.
        
        Returns
        -------
        FitResult
            Parameters, fitness and diagnostics in participant-ID order. Restarts use
            the same keys as serial participant fitting.
        """
        started = time.perf_counter()
        subjects, trial_indices = self._subject_selections(trial_mask)
        combined = self._empty_result()
        if len(subjects):
            outcomes = self._subject_restarts(trial_indices)
            for subject, outcome in zip(subjects, outcomes, strict=True):
                result = self._format_result(outcome, int(subject), started)
                self._append_subject_result(combined, result)
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def _subject_selections(
        self, trial_mask: Bool[Array, " trials"]
    ) -> tuple[np.ndarray, Array]:
        """Select nonempty subjects and construct a rectangular trial-index matrix."""
        masks, subjects = make_subject_trial_masks(trial_mask, self.subjects)
        if not masks:
            return subjects, jnp.empty((0, 0), dtype=jnp.int32)
        indices = [np.where(mask)[0] for mask in masks]
        return subjects, jnp.asarray(indices)

    def _subject_restart_keys(
        self, subject_count: int
    ) -> UInt32[Array, " participants restarts 2"]:
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

    def _subject_restarts(
        self,
        trial_indices: Integer[Array, "participants trials"],
    ) -> list[list[_RestartResult]]:
        """Run each restart across subjects, preserving row-aligned outcomes."""
        keys = self._subject_restart_keys(len(trial_indices))
        outcomes: list[list[_RestartResult]] = [[] for _ in trial_indices]
        for restart in trange(
            self.all_hyperparams["best_of"],
            disable=not self.progress_bar or self.all_hyperparams["best_of"] <= 1,
        ):
            raw = self._compiled_subject_runs(keys[:, restart, :], trial_indices)
            for subject, (fitness, parameters, generations, converged) in enumerate(
                zip(*raw, strict=True)
            ):
                outcomes[subject].append(
                    _RestartResult(
                        float(fitness),
                        np.asarray(parameters, dtype=float),
                        int(generations),
                        bool(converged),
                    )
                )
        return outcomes

    def _run_subjects(
        self,
        subject_keys: UInt32[Array, " participants 2"],
        subject_trial_indices: Integer[Array, " participants trials"],
    ) -> _RunArrays:
        """Scan ordinary-loss DE runs over equally sized trial-index selections."""

        def step(
            carry: None,
            selection: tuple[PRNGKeyArray, Integer[Array, " trials"]],
        ) -> tuple[None, _RunArrays]:
            key, indices = selection
            return carry, self._run_selected_trials(key, indices)

        _, outputs = jax.lax.scan(step, None, (subject_keys, subject_trial_indices))
        return outputs


class MapEvosaxDE(ScanEvosaxDE):
    """Use JAX map for equal-length subject fits, with optional subject batching."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Parameters,
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: LossFactory,
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
        self,
        subject_keys: UInt32[Array, " participants 2"],
        subject_trial_indices: Integer[Array, " participants trials"],
    ) -> _RunArrays:
        """Map ordinary-loss DE runs using the configured subject batch size."""

        def fit_subject(
            selection: tuple[PRNGKeyArray, Integer[Array, " trials"]],
        ) -> _RunArrays:
            key, indices = selection
            return self._run_selected_trials(key, indices)

        return jax.lax.map(
            fit_subject,
            (subject_keys, subject_trial_indices),
            batch_size=self.map_batch_size,
        )
