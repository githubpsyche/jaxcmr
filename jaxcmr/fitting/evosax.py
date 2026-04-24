"""Evosax differential-evolution fitting."""

import time
from typing import Any, Mapping, Optional, Type

import jax
import numpy as np
from jax import numpy as jnp
from tqdm import trange

from evosax.algorithms import DifferentialEvolution
from evosax.core.restart import FitnessStdRestartParams, fitness_std_cond

from jaxcmr.helpers import make_subject_trial_masks
from jaxcmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    Float_,
    Integer,
    LossFn,
    MemorySearchModelFactory,
    RecallDataset,
)


__all__ = [
    "BatchEvosaxDE",
    "EvosaxDE",
    "MapEvosaxDE",
    "ScanEvosaxDE",
]


class EvosaxDE:
    """A fitting class that uses evosax's Differential Evolution algorithm."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_pool_items features_count"]],
        base_params: Mapping[str, Float_],
        model_factory_cls: Type[MemorySearchModelFactory],
        loss_fn_cls: Type[LossFn],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        """
        Configure the fitting algorithm.

        Args:
            dataset: The dataset containing trial data (including 'subject').
            features: Optional feature matrix aligned to the vocabulary.
            base_params: A dictionary of parameters that are held fixed.
            model_factory_cls: Class implementing MemorySearchModelFactory.
            loss_fn_cls: Class implementing LossFn.
            hyperparams: Optional dictionary of hyperparameters for the fitting routine.
                May include 'bounds' (dict[str, list[float]]) and other keys
                like 'num_steps', 'pop_size', etc.
        """
        # Store essential data
        self.dataset = dataset
        self.base_params = base_params
        self.subjects = dataset["subject"].flatten()

        # configure convenience features
        if hyperparams is None:
            hyperparams = {}
        self.progress_bar = hyperparams.get("progress_bar", True)
        self.display_iterations = hyperparams.get("display_iterations", False)

        # Extract bounds for free params; store all hyperparams for convenience
        self.free_parameter_bounds = hyperparams.get("bounds", {})
        self.bounds = np.array(list(self.free_parameter_bounds.values()))
        self.lower_bounds = jnp.asarray(self.bounds[:, 0])
        self.upper_bounds = jnp.asarray(self.bounds[:, 1])
        self.num_parameters = len(self.free_parameter_bounds)

        pop_size = hyperparams.get("pop_size", 15)
        diff_w = hyperparams.get("diff_w", (0.5, 1.0))
        if not isinstance(diff_w, (tuple, list)):
            diff_w = (diff_w, diff_w)
        diff_w = tuple(diff_w)
        self.all_hyperparams = {
            "bounds": self.free_parameter_bounds,
            "num_steps": hyperparams.get("num_steps", 1000),
            "pop_size": pop_size,
            "population_size": max(4, int(pop_size) * self.num_parameters),
            "relative_tolerance": hyperparams.get("relative_tolerance", 0.001),
            "absolute_tolerance": hyperparams.get("absolute_tolerance", 0.0),
            "cross_over_rate": hyperparams.get("cross_over_rate", 0.9),
            "diff_w": diff_w,
            "best_of": hyperparams.get("best_of", 1),
            "init": hyperparams.get("init", "random"),
            "boundary_handling": hyperparams.get("boundary_handling", "resample"),
            "seed": int(hyperparams.get("seed", 0)),
            "nonfinite_penalty": float(hyperparams.get("nonfinite_penalty", 1e30)),
        }
        self._restart_params = FitnessStdRestartParams(
            tol=self.all_hyperparams["relative_tolerance"],
            atol=self.all_hyperparams["absolute_tolerance"],
        )
        self._base_key = jax.random.PRNGKey(self.all_hyperparams["seed"])
        self._fit_counter = 0

        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)
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
        self._compiled_run_scan = jax.jit(self._run_scan)

    def _next_key(self) -> jax.Array:
        """Return a deterministic key for the next public fit call."""
        key = jax.random.fold_in(self._base_key, self._fit_counter)
        self._fit_counter += 1
        return key

    def _sample_initial_population(self, key: jax.Array) -> jax.Array:
        """Sample an initial bounded population."""
        if self.all_hyperparams["init"] == "latinhypercube":
            sample_key, permutation_key = jax.random.split(key)
            segment_size = 1.0 / self.all_hyperparams["population_size"]
            samples = (
                segment_size
                * jax.random.uniform(
                    sample_key,
                    (
                        self.all_hyperparams["population_size"],
                        self.num_parameters,
                    ),
                )
                + jnp.linspace(
                    0.0,
                    1.0,
                    self.all_hyperparams["population_size"],
                    endpoint=False,
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

    def _ensure_bounds(self, key: jax.Array, population: jax.Array) -> jax.Array:
        """Repair out-of-bounds trial values."""
        out_of_bounds = jnp.logical_or(
            population < self.lower_bounds, population > self.upper_bounds
        )
        if self.all_hyperparams["boundary_handling"] == "resample":
            replacement = self.lower_bounds + jax.random.uniform(
                key, population.shape, dtype=population.dtype
            ) * (self.upper_bounds - self.lower_bounds)
            return jnp.where(out_of_bounds, replacement, population)
        return jnp.clip(population, self.lower_bounds, self.upper_bounds)

    def _evaluate_trial_population(
        self,
        trial_indices: jax.Array,
        population: jax.Array,
    ) -> jax.Array:
        """Evaluate a candidate population with dynamic trial indices."""
        fitness = jnp.asarray(
            self.loss_fn(
                trial_indices,
                self.base_params,
                self.free_parameter_bounds,
                population.T,
            )
        )
        fitness = jnp.reshape(fitness, (population.shape[0],))
        return jnp.where(
            jnp.isfinite(fitness),
            fitness,
            jnp.asarray(self.all_hyperparams["nonfinite_penalty"]),
        )

    def _has_converged(self, fitness: jax.Array) -> jax.Array:
        """Return whether population fitness meets the SciPy DE criterion."""
        penalty = jnp.asarray(
            self.all_hyperparams["nonfinite_penalty"], dtype=fitness.dtype
        )
        not_penalized = jnp.all(fitness != penalty)
        return jnp.logical_and(
            not_penalized,
            fitness_std_cond(None, fitness, None, None, None, self._restart_params),
        )

    def _run_scan(
        self,
        key: jax.Array,
        trial_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run the compiled evosax DE scan."""
        num_steps = int(self.all_hyperparams["num_steps"])
        init_key, tell_key, loop_key = jax.random.split(key, 3)
        population = self._sample_initial_population(init_key)
        fitness = self._evaluate_trial_population(trial_indices, population)
        state = self._algorithm.init(
            init_key, population, fitness, self._algorithm_params
        )
        state, _ = self._algorithm.tell(
            tell_key, population, fitness, state, self._algorithm_params
        )

        def step(carry, _):
            def no_op_step(carry):
                return carry

            def evolve_step(carry):
                loop_key, state, _, nit = carry
                loop_key, ask_key, bounds_key, tell_key = jax.random.split(
                    loop_key, 4
                )
                population, state = self._algorithm.ask(
                    ask_key, state, self._algorithm_params
                )
                population = self._ensure_bounds(bounds_key, population)
                fitness = self._evaluate_trial_population(trial_indices, population)
                state, _ = self._algorithm.tell(
                    tell_key, population, fitness, state, self._algorithm_params
                )
                converged = self._has_converged(state.fitness)
                return (loop_key, state, converged, nit + 1)

            _, _, converged, _ = carry
            carry = jax.lax.cond(converged, no_op_step, evolve_step, carry)
            return carry, None

        (_, state, converged, nit), _ = jax.lax.scan(
            step,
            (loop_key, state, jnp.asarray(False), jnp.asarray(0)),
            xs=None,
            length=num_steps,
        )
        return (
            state.best_fitness,
            self._algorithm.get_best_solution(state),
            nit,
            converged,
        )

    def _run_once(
        self,
        key: jax.Array,
        trial_indices: Integer[Array, " trials"],
    ) -> tuple[float, np.ndarray, int, bool]:
        """Run one evosax DE optimization and return best fitness/parameters."""
        best_fitness, best_params, nit, converged = self._compiled_run_scan(
            key, trial_indices
        )
        best_fitness = float(best_fitness)
        best_params = np.asarray(best_params, dtype=float)
        return best_fitness, best_params, int(nit), bool(converged)

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit one parameter set to the trials selected by the mask.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Boolean mask selecting which trials to include in the fit.
        subject_id : int, optional
            Label stored in the result dict. Defaults to -1.

        Returns
        -------
        FitResult
            Fitted parameters, fitness value, and metadata.
        """
        t0 = time.perf_counter()

        # Convert the mask to an array of trial indices
        trial_indices = jnp.where(trial_mask)[0]

        # Run differential evolution
        best_fitness = np.inf
        best_params = None
        best_nit = 0
        best_converged = False
        fit_key = self._next_key()
        run_keys = jax.random.split(fit_key, self.all_hyperparams["best_of"])
        for run_key in run_keys:
            run_fitness, run_params, run_nit, run_converged = self._run_once(
                run_key, trial_indices
            )
            if run_fitness < best_fitness:
                best_fitness = run_fitness
                best_params = run_params
                best_nit = run_nit
                best_converged = run_converged

        assert best_params is not None, "No fit result found"
        return {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [float(best_fitness)],
            "fits": {
                # For each base param, we just repeat its original value
                **{k: [float(v)] for k, v in self.base_params.items()},
                # For each free param, we store the optimizer's best value
                **{
                    # Map each free param name to the best-fit value
                    param_name: [float(best_params[idx])]
                    for idx, param_name in enumerate(self.free_parameter_bounds)
                },
                "subject": [subject_id],
            },
            "nit": [best_nit],
            "converged": [best_converged],
            "hyperparameters": self.all_hyperparams,
            "fit_time": time.perf_counter() - t0,
        }

    def fit_subjects(
        self, trial_mask: Bool[Array, " trials"]
    ) -> FitResult:
        """Fit each subject independently and accumulate results.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Boolean mask selecting which trials to include.

        Returns
        -------
        FitResult
            Combined results with per-subject fitted parameters.
        """
        t0 = time.perf_counter()

        subject_trial_masks, unique_subjects = make_subject_trial_masks(
            trial_mask, self.subjects
        )

        # Prepare a global result structure
        all_results: FitResult = {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [],
            "nit": [],
            "converged": [],
            "fits": {
                **{k: [] for k in self.free_parameter_bounds},
                **{k: [] for k in self.base_params},
                "subject": [],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": 0.0,
        }

        # Optionally show progress bar
        subject_range = (
            trange(len(unique_subjects))
            if self.progress_bar
            else range(len(unique_subjects))
        )
        for s in subject_range:
            # If no trials for this subject, skip
            if np.sum(subject_trial_masks[s]) == 0:
                continue

            fit_result = self.fit(
                subject_trial_masks[s], int(unique_subjects[s])
            )
            all_results["fitness"] += fit_result["fitness"]
            all_results["nit"] += fit_result["nit"]
            all_results["converged"] += fit_result["converged"]

            # Show in tqdm progress bar
            if self.progress_bar:
                subject_range.set_description(  # type: ignore
                    f"Subject={unique_subjects[s]}, Fitness={fit_result['fitness'][0]}"
                )

            # Accumulate fitted parameters
            all_results["fits"]["subject"].append(int(unique_subjects[s]))

            # Append param values
            for param_name, values_list in fit_result["fits"].items():
                # Skip 'subject' to avoid double-appending
                if param_name == "subject":
                    continue
                all_results["fits"][param_name] += values_list

            # Bail out if we got a non-finite fitness
            if not jnp.isfinite(fit_result["fitness"][0]):
                raise ValueError(
                    f"Non-finite fitness for subject {unique_subjects[s]}", fit_result
                )

        all_results["fit_time"] = time.perf_counter() - t0
        return all_results


class ScanEvosaxDE(EvosaxDE):
    """A fitting class that scans equal-length subject fits with evosax DE."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_pool_items features_count"]],
        base_params: Mapping[str, Float_],
        model_factory_cls: Type[MemorySearchModelFactory],
        loss_fn_cls: Type[LossFn],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            dataset,
            features,
            base_params,
            model_factory_cls,
            loss_fn_cls,
            hyperparams=hyperparams,
        )
        self._compiled_run_batch = jax.jit(self._run_batch)

    def _run_batch(
        self,
        subject_keys: jax.Array,
        subject_trial_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run the compiled evosax DE scan across multiple subjects."""

        def step(carry, xs):
            key, trial_indices = xs
            return carry, self._run_scan(key, trial_indices)

        _, outputs = jax.lax.scan(
            step,
            None,
            (subject_keys, subject_trial_indices),
        )
        return outputs

    def fit_subjects(
        self, trial_mask: Bool[Array, " trials"]
    ) -> FitResult:
        """Fit each subject independently and accumulate results."""
        t0 = time.perf_counter()

        subject_trial_masks, unique_subjects = make_subject_trial_masks(
            trial_mask, self.subjects
        )

        all_results: FitResult = {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [],
            "nit": [],
            "converged": [],
            "fits": {
                **{k: [] for k in self.free_parameter_bounds},
                **{k: [] for k in self.base_params},
                "subject": [],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": 0.0,
        }

        valid_subject_indices = [
            s for s in range(len(unique_subjects)) if np.sum(subject_trial_masks[s]) != 0
        ]
        if len(valid_subject_indices) == 0:
            all_results["fit_time"] = time.perf_counter() - t0
            return all_results

        subject_trial_indices = [
            np.where(subject_trial_masks[s])[0] for s in valid_subject_indices
        ]
        trial_counts = {len(indices) for indices in subject_trial_indices}
        if len(trial_counts) != 1:
            raise ValueError(
                "ScanEvosaxDE.fit_subjects requires the same number of selected trials per subject."
            )

        fit_counter = self._fit_counter
        subject_trial_indices = jnp.asarray(subject_trial_indices)
        fit_keys = jnp.asarray(
            [
                jax.random.fold_in(self._base_key, fit_counter + s)
                for s in range(len(valid_subject_indices))
            ]
        )
        self._fit_counter += len(valid_subject_indices)

        run_keys = jax.vmap(
            lambda key: jax.random.split(key, self.all_hyperparams["best_of"])
        )(fit_keys)

        run_fitness, run_params, run_nit, run_converged = self._compiled_run_batch(
            run_keys[:, 0, :], subject_trial_indices
        )
        best_fitness = np.asarray(run_fitness, dtype=float)
        best_params = np.asarray(run_params, dtype=float)
        best_nit = np.asarray(run_nit, dtype=int)
        best_converged = np.asarray(run_converged, dtype=bool)

        start_range = (
            trange(1, self.all_hyperparams["best_of"])
            if self.progress_bar and self.all_hyperparams["best_of"] > 1
            else range(1, self.all_hyperparams["best_of"])
        )
        for run_index in start_range:
            run_fitness, run_params, run_nit, run_converged = (
                self._compiled_run_batch(
                    run_keys[:, run_index, :], subject_trial_indices
                )
            )
            run_fitness = np.asarray(run_fitness, dtype=float)
            run_params = np.asarray(run_params, dtype=float)
            run_nit = np.asarray(run_nit, dtype=int)
            run_converged = np.asarray(run_converged, dtype=bool)
            improved = run_fitness < best_fitness
            best_fitness = np.where(improved, run_fitness, best_fitness)
            best_params = np.where(improved[:, None], run_params, best_params)
            best_nit = np.where(improved, run_nit, best_nit)
            best_converged = np.where(improved, run_converged, best_converged)

        for result_index, subject_index in enumerate(valid_subject_indices):
            all_results["fitness"].append(float(best_fitness[result_index]))
            all_results["nit"].append(int(best_nit[result_index]))
            all_results["converged"].append(bool(best_converged[result_index]))
            all_results["fits"]["subject"].append(int(unique_subjects[subject_index]))

            for param_name, param_index in zip(
                self.free_parameter_bounds, range(best_params.shape[1])
            ):
                all_results["fits"][param_name].append(
                    float(best_params[result_index, param_index])
                )
            for param_name, param_value in self.base_params.items():
                all_results["fits"][param_name].append(float(param_value))

            if not np.isfinite(best_fitness[result_index]):
                raise ValueError(
                    f"Non-finite fitness for subject {unique_subjects[subject_index]}",
                    {
                        "fitness": float(best_fitness[result_index]),
                        "subject": int(unique_subjects[subject_index]),
                    },
                )

        all_results["fit_time"] = time.perf_counter() - t0
        return all_results


class MapEvosaxDE(ScanEvosaxDE):
    """A fitting class that maps equal-length subject fits with evosax DE."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_pool_items features_count"]],
        base_params: Mapping[str, Float_],
        model_factory_cls: Type[MemorySearchModelFactory],
        loss_fn_cls: Type[LossFn],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        if hyperparams is None:
            hyperparams = {}
        self.map_batch_size = hyperparams.get("map_batch_size", None)
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

    def _run_batch(
        self,
        subject_keys: jax.Array,
        subject_trial_indices: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run the compiled evosax DE path with lax.map over subjects."""

        def fit_subject(xs):
            key, trial_indices = xs
            return self._run_scan(key, trial_indices)

        return jax.lax.map(
            fit_subject,
            (subject_keys, subject_trial_indices),
            batch_size=self.map_batch_size,
        )


# Compatibility aliases. Do not use in new code.
BatchEvosaxDE = ScanEvosaxDE
