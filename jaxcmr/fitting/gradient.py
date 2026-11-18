"""Bounded SciPy fitting using compiled JAX values and gradients.

Each fit selects its trials, screens candidate starts, runs independent local
restarts, and reports the best valid evaluation. SciPy and deadline accounting
run on the host; parameter conversion, loss evaluation and gradients use JAX.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import BFGS, Bounds, OptimizeResult, minimize
from scipy.stats import qmc

from jaxcmr.dataset import make_subject_trial_masks
from jaxcmr.fitting.initialization import select_starts
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
    RecallDataset,
)

__all__ = ["ScipyLBFGSB", "ScipyTrustConstr"]

type _ValueAndGradientFn = Callable[[Array], tuple[Array, Array]]
type _ScreenValuesFn = Callable[[Array], Array]
type _ScipyObjective = Callable[[np.ndarray], tuple[float, np.ndarray]]


class _NonfiniteEvaluation(ValueError):
    """End a restart whose objective or gradient cannot define a valid fit."""


class _BudgetExpired(Exception):
    """Stop SciPy between evaluations when the current time allowance expires."""


@dataclass
class _FitProgress:
    """Host-side results shared by the restarts of one fit, never JAX state.

    A restart owns its diagnostics; this record owns the best evaluation and
    improvement history across restarts. Only deadline-eligible evaluations
    are offered to consider(), after checking both the value and its gradient.
    """

    started: float
    deadline: float
    best_value: float = np.inf
    best_unit: np.ndarray | None = None
    best_restart: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def consider(
        self,
        value: float,
        unit: Float[np.ndarray, " parameters"],
        restart: int,
        observed: float,
    ) -> None:
        """Retain a strict improvement, copying coordinates owned by SciPy."""
        if value >= self.best_value:
            return
        self.best_value = value
        self.best_unit = np.asarray(unit).copy()
        self.best_restart = restart
        self.history.append(
            {
                "seconds": observed - self.started,
                "fitness": value,
                "unit": self.best_unit.tolist(),
                "restart": restart,
            }
        )


class _GradientOptimizer:
    """Shared screening, restart, deadline and result-selection machinery.

    Dataset-facing and prepared fitters supply value/gradient and screening
    functions; this class does not own observations or select trials.
    """

    _method: str

    def _configure_optimizer(
        self,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None,
    ) -> None:
        """Configure coordinates and solver controls independently of the objective."""
        options = dict(hyperparams or {})
        self.base_params = dict(base_params)
        self.free_parameter_bounds = {
            name: list(pair) for name, pair in options.get("bounds", {}).items()
        }
        self.parameter_space = ParameterSpace(
            self.free_parameter_bounds, options.get("transforms")
        )
        self.sampling_space = ParameterSpace(
            self.free_parameter_bounds,
            options.get("sampling_transforms", self.parameter_space.transforms),
        )
        self.names = self.parameter_space.names
        self.bounds = self.parameter_space.bounds
        self.all_hyperparams = self._configure_hyperparameters(options)
        self._initial = self._initial_parameters()
        self._fit_counter = 0
        self._lower = self.parameter_space.lower
        self._upper = self.parameter_space.upper
        self._span = self._upper - self._lower
        self._physical_parameters = jax.jit(self.parameter_space.to_physical)
        self._unit_parameters = jax.jit(self.parameter_space.to_unit)
        self._sampling_parameters = jax.jit(self.sampling_space.to_unit)
        self._sampling_to_unit = jax.jit(
            lambda unit: self.parameter_space.to_unit(
                self.sampling_space.to_physical(unit)
            )
        )

    def _fit_objective(
        self,
        value_and_gradient: _ValueAndGradientFn,
        screen_values: _ScreenValuesFn,
        subject_id: int,
        started: float,
    ) -> FitResult:
        """Screen and optimize an objective already bound to its observations."""
        config = self.all_hyperparams
        rng = np.random.default_rng([int(config["seed"]), self._fit_counter])
        self._fit_counter += 1
        limit = config["time_limit_seconds"]
        progress = _FitProgress(started, np.inf if limit is None else started + limit)
        units, selected, screening = self._starting_points(screen_values, rng, progress)
        for restart, candidate_index in enumerate(selected):
            if progress.records and time.perf_counter() >= progress.deadline:
                break
            record = self._run_restart(
                value_and_gradient,
                units[candidate_index],
                restart,
                int(candidate_index),
                progress,
            )
            progress.records.append(record)
        return self._format_result(progress, screening, subject_id)

    def _warmup_objective(
        self,
        value_and_gradient: _ValueAndGradientFn,
        screen_values: _ScreenValuesFn,
    ) -> None:
        """Warm objective and coordinate calculations without consuming restart keys."""
        point = jnp.full(len(self.names), 0.5, dtype=self._lower.dtype)
        jax.block_until_ready(value_and_gradient(point))
        self._physical_parameters(point).block_until_ready()
        self._unit_parameters(self._physical_parameters(point)).block_until_ready()
        self._sampling_parameters(self._physical_parameters(point)).block_until_ready()
        count = (
            self.all_hyperparams["screening_size"] or self.all_hyperparams["best_of"]
        )
        self._sampling_to_unit(jnp.tile(point, (count, 1))).block_until_ready()
        if self.all_hyperparams["screening_size"]:
            batch = jnp.tile(point, (self.all_hyperparams["screening_batch_size"], 1))
            screen_values(batch).block_until_ready()

    def _configure_hyperparameters(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve defaults and retain the public option names in result metadata."""
        trust = self._method == "trust-constr"
        config = {
            "bounds": self.free_parameter_bounds,
            "transforms": self.parameter_space.transforms,
            "sampling_transforms": self.sampling_space.transforms,
            "method": self._method,
            "best_of": options.get("best_of", 4 if trust else 8),
            "screening_size": options.get("screening_size", 64),
            "screening_batch_size": options.get("screening_batch_size", 16),
            "min_start_distance": options.get("min_start_distance", 0.1),
            "seed": options.get("seed", 0),
            "num_steps": options.get("num_steps", 1000 if trust else 500),
            "max_fun": options.get("max_fun", 15000),
            "relative_tolerance": options.get("relative_tolerance", 1e-9),
            "gradient_tolerance": options.get(
                "gradient_tolerance", 1e-7 if trust else 1e-5
            ),
            "max_line_search_steps": options.get("max_line_search_steps", 30),
            "time_limit_seconds": options.get("time_limit_seconds"),
            "local_time_limit_seconds": options.get("local_time_limit_seconds"),
            "initial_params": options.get("initial_params"),
        }
        if trust:
            for key in ("max_fun", "relative_tolerance", "max_line_search_steps"):
                del config[key]
            config.update(
                step_tolerance=options.get("step_tolerance", 1e-12),
                barrier_tolerance=options.get("barrier_tolerance", 1e-10),
                initial_trust_radius=options.get("initial_trust_radius", 1.0),
                initial_barrier_parameter=options.get("initial_barrier_parameter", 0.1),
            )
        return config

    def _initial_parameters(self) -> np.ndarray | None:
        """Order an optional complete starting vector in physical parameter units."""
        initial = self.all_hyperparams["initial_params"]
        if initial is not None:
            initial = np.array([initial[name] for name in self.names], dtype=float)
        return initial

    def _starting_points(
        self,
        screen_values: _ScreenValuesFn,
        rng: np.random.Generator,
        progress: _FitProgress,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Draw and optionally screen starts in sampling coordinates.

        Returns all candidates in optimization coordinates, the selected row
        indices, and screening diagnostics. Conversion follows screening so
        distances retain their specified meaning in the sampling space.
        """
        config = self.all_hyperparams
        size = config["screening_size"]
        if size:
            candidates = qmc.LatinHypercube(d=len(self.names), rng=rng).random(size)
        else:
            candidates = rng.uniform(size=(config["best_of"], len(self.names)))
        if self._initial is not None:
            candidates[0] = np.asarray(
                self._sampling_parameters(
                    jnp.asarray(self._initial, dtype=self._lower.dtype)
                )
            )
        screening: dict[str, Any] = {"requested": size, "evaluated": 0}
        if size:
            scores, screening["evaluated"] = self._screen_candidates(
                candidates, screen_values, progress.deadline
            )
            # An explicit initial point is a request for local refinement, not
            # a candidate that screening may silently discard.
            ranked_scores = scores.copy()
            if self._initial is not None:
                ranked_scores[0] = -np.finfo(float).max
            selected = select_starts(
                candidates,
                ranked_scores,
                config["best_of"],
                config["min_start_distance"],
            )
            screening.update(
                candidate_unit=candidates.tolist(),
                coordinates="sampling",
                scores=[float(v) if np.isfinite(v) else None for v in scores],
                selected_indices=selected.tolist(),
                seconds=time.perf_counter() - progress.started,
            )
        else:
            selected = np.arange(len(candidates))
        units = np.asarray(
            self._sampling_to_unit(jnp.asarray(candidates, dtype=self._lower.dtype))
        )
        return units, selected, screening

    def _screen_candidates(
        self,
        candidates: Float[np.ndarray, " candidates parameters"],
        screen_values: _ScreenValuesFn,
        deadline: float,
    ) -> tuple[Float[np.ndarray, " candidates"], int]:
        """Score fixed-size batches, leaving unevaluated candidates as NaN.

        One batch is always evaluated, even if compilation exceeds the budget.
        Padding keeps the last batch's compiled shape stable; its extra scores
        are discarded before ranking candidates.
        """
        scores = np.full(len(candidates), np.nan)
        evaluated = 0
        batch_size = self.all_hyperparams["screening_batch_size"]
        for start in range(0, len(candidates), batch_size):
            if start and time.perf_counter() >= deadline:
                break
            batch = candidates[start : start + batch_size]
            padded = np.pad(batch, ((0, batch_size - len(batch)), (0, 0)), mode="edge")
            scores[start : start + len(batch)] = np.asarray(
                screen_values(jnp.asarray(padded, dtype=self._lower.dtype))
            )[: len(batch)]
            evaluated += len(batch)
        return scores, evaluated

    def _run_restart(
        self,
        value_and_gradient: _ValueAndGradientFn,
        initial: Float[np.ndarray, " parameters"],
        restart: int,
        candidate_index: int,
        progress: _FitProgress,
    ) -> dict[str, Any]:
        """Run one local solve with its own diagnostics and callback scope.

        No callback captures variables from the outer restart loop. A later
        restart cannot change this one's record, identifier or local deadline.
        """
        record: dict[str, Any] = {
            "restart": restart,
            "initial_unit": initial.tolist(),
            "candidate_index": candidate_index,
            "nfev": 0,
            "nit": 0,
            "converged": False,
            "fitness": None,
            "parameters": None,
        }
        started = time.perf_counter()
        local_limit = self.all_hyperparams["local_time_limit_seconds"]
        deadline = (
            min(progress.deadline, started + local_limit)
            if local_limit is not None
            else progress.deadline
        )
        evaluate = self._restart_objective(
            value_and_gradient, record, progress, deadline
        )

        def callback(
            _unit: Float[np.ndarray, " parameters"],
            _state: OptimizeResult | None = None,
        ) -> None:
            """Count iterations even when an exception interrupts SciPy."""
            record["nit"] += 1

        try:
            result = self._minimize(evaluate, initial, callback)
            self._record_solver_result(record, result)
        except _BudgetExpired:
            record["budget"] = (
                "global" if time.perf_counter() >= progress.deadline else "local"
            )
            record["message"] = (
                f"Time budget exhausted between evaluations ({record['budget']})."
            )
        except _NonfiniteEvaluation as error:
            record["message"] = str(error)
        record["seconds"] = time.perf_counter() - started
        return record

    def _restart_objective(
        self,
        value_and_gradient: _ValueAndGradientFn,
        record: dict[str, Any],
        progress: _FitProgress,
        deadline: float,
    ) -> _ScipyObjective:
        """Bind one restart's objective, enforcing deadlines around each evaluation.

        The first fit evaluation is a fallback even if it finishes late. Later
        evaluations must finish within both budgets to update the reported best.
        Invalid gradients reject a candidate even when its loss is finite.
        """

        def evaluate(
            unit: Float[np.ndarray, " parameters"],
        ) -> tuple[float, Float[np.ndarray, " parameters"]]:
            initial_fallback = record["nfev"] == 0 and record["restart"] == 0
            if not initial_fallback and time.perf_counter() >= deadline:
                raise _BudgetExpired
            value, gradient = value_and_gradient(
                jnp.asarray(unit, dtype=self._lower.dtype)
            )
            value, gradient = float(value), np.asarray(gradient, dtype=float)
            record["nfev"] += 1
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                record["invalid_unit"] = np.asarray(unit).tolist()
                raise _NonfiniteEvaluation(
                    "Nonfinite loss or gradient; inspect the model/loss derivatives."
                )
            observed = time.perf_counter()
            if observed > deadline:
                record["late_evaluations"] = record.get("late_evaluations", 0) + 1
                if initial_fallback:
                    record["used_initial_fallback"] = True
            if observed <= deadline or initial_fallback:
                if record["fitness"] is None or value < record["fitness"]:
                    record["fitness"] = value
                    record["best_unit"] = np.asarray(unit).tolist()
                    physical = self._physical_parameters(
                        jnp.asarray(unit, dtype=self._lower.dtype)
                    )
                    record["parameters"] = dict(
                        zip(self.names, map(float, physical), strict=True)
                    )
                progress.consider(value, unit, record["restart"], observed)
            return value, gradient

        return evaluate

    def _minimize(
        self,
        evaluate: _ScipyObjective,
        initial: Float[np.ndarray, " parameters"],
        callback: Callable[..., None],
    ) -> OptimizeResult:
        """Translate public options and run SciPy in bounded unit coordinates."""
        config = self.all_hyperparams
        solver_options = {
            "maxiter": config["num_steps"],
            "gtol": config["gradient_tolerance"],
        }
        if self._method == "trust-constr":
            solver_options.update(
                xtol=config["step_tolerance"],
                barrier_tol=config["barrier_tolerance"],
                initial_tr_radius=config["initial_trust_radius"],
                initial_barrier_parameter=config["initial_barrier_parameter"],
            )
            # Each restart gets a fresh curvature estimate; no search state is shared.
            solver_extra = {"hess": BFGS()}
        else:
            solver_options.update(
                maxfun=config["max_fun"],
                ftol=config["relative_tolerance"],
                maxls=config["max_line_search_steps"],
            )
            solver_extra = {}
        return minimize(
            evaluate,
            initial,
            jac=True,
            method=self._method,
            bounds=Bounds(0.0, 1.0, keep_feasible=True),
            callback=callback,
            options=solver_options,
            **solver_extra,
        )

    def _record_solver_result(
        self, record: dict[str, Any], result: OptimizeResult
    ) -> None:
        """Record solver termination and the bound-aware objective-gradient norm."""
        record.update(
            nit=int(result.nit),
            solver_converged=bool(result.success),
            converged=bool(result.success)
            and np.array_equal(record.get("best_unit", []), result.x),
            message=str(result.message),
            final_fitness=float(result.fun),
            final_unit=result.x.tolist(),
            final_parameters=dict(
                zip(
                    self.names,
                    map(
                        float,
                        self._physical_parameters(
                            jnp.asarray(result.x, dtype=self._lower.dtype)
                        ),
                    ),
                    strict=True,
                )
            ),
        )
        gradient = result.grad if self._method == "trust-constr" else result.jac
        # Ignore components whose downhill direction leaves an active bound.
        projected = np.where(
            ((result.x <= 0) & (gradient > 0)) | ((result.x >= 1) & (gradient < 0)),
            0.0,
            gradient,
        )
        record["projected_gradient_norm"] = float(np.max(np.abs(projected)))
        if self._method == "trust-constr":
            record.update(
                optimality=float(result.optimality),
                constraint_violation=float(result.constr_violation),
            )

    def _format_result(
        self, progress: _FitProgress, screening: dict[str, Any], subject_id: int
    ) -> FitResult:
        """Report the best eligible evaluation in the common fitter result schema.

        Solver convergence belongs to its final point. A better earlier point
        is returned when appropriate, without borrowing that convergence flag.
        """
        if progress.best_unit is None or progress.best_restart is None:
            raise ValueError(
                f"No finite loss/gradient pair was found. Restart diagnostics: {progress.records}"
            )
        chosen = progress.records[progress.best_restart]
        # Return the parameters evaluated in JAX's configured precision, rather
        # than a different double-precision reconstruction on the host.
        params = np.asarray(
            self._physical_parameters(
                jnp.asarray(progress.best_unit, dtype=self._lower.dtype)
            )
        )
        final_unit = chosen.get("final_unit")
        converged = (
            chosen["converged"]
            and final_unit is not None
            and np.array_equal(progress.best_unit, final_unit)
        )
        return {
            "fixed": {
                key: np.asarray(value).item() for key, value in self.base_params.items()
            },
            "free": self.free_parameter_bounds,
            "fitness": [progress.best_value],
            "nit": [chosen["nit"]],
            "converged": [bool(converged)],
            "fits": {
                **{
                    key: [np.asarray(value).item()]
                    for key, value in self.base_params.items()
                },
                **{key: [float(params[i])] for i, key in enumerate(self.names)},
                "subject": [subject_id],
            },
            "hyperparameters": dict(self.all_hyperparams),
            "fit_time": time.perf_counter() - progress.started,
            "diagnostics": [
                {
                    "subject": subject_id,
                    "restarts": progress.records,
                    "selected_restart": progress.best_restart,
                    "screening": screening,
                    "history": [
                        dict(
                            row,
                            parameters=dict(
                                zip(
                                    self.names,
                                    np.asarray(
                                        self.parameter_space.to_physical(
                                            jnp.asarray(
                                                row["unit"], dtype=self._lower.dtype
                                            )
                                        )
                                    ).tolist(),
                                    strict=True,
                                )
                            ),
                        )
                        for row in progress.history
                    ],
                    "nfev": sum(record["nfev"] for record in progress.records),
                    "jax_dtype": str(self._lower.dtype),
                    "effective_bounds": np.stack(
                        [np.asarray(self._lower), np.asarray(self._upper)], axis=1
                    ).tolist(),
                }
            ],
        }


class _ScipyGradient(_GradientOptimizer):
    """Bind ordinary trial-index losses to the shared gradient optimizer."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Parameters,
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: LossFactory,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure parameter coordinates, solver controls and compiled losses.

        Parameters
        ----------
        dataset : RecallDataset
            Observations with one subject label per trial.
        features : Array or None
            Optional vocabulary features passed to the loss constructor.
        base_params : Parameters
            Model defaults and fixed values; free parameters override defaults.
        model_factory_cls : type[MemorySearchModelFactory]
            Factory passed to the loss constructor.
        loss_fn_cls : LossFactory
            Differentiable population loss.
        hyperparams : dict[str, Any] or None
            Bounds and fitting controls documented on this class and its solver.
        """
        self.dataset = dataset
        self.subjects = np.asarray(dataset["subject"]).reshape(-1)
        self._configure_optimizer(base_params, hyperparams)
        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)
        self._value_and_grad, self._screen_values = self._make_loss_functions()

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit selected trials with screening and independent local restarts.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Nonempty selection with one Boolean entry per dataset trial.
        subject_id : int, optional
            Subject label stored in the result, defaulting to -1.

        Returns
        -------
        FitResult
            Best eligible evaluation, physical parameters, convergence and
            per-restart diagnostics. Elapsed time includes trial selection,
            compilation, screening and every restart unless warmed beforehand.
        """
        started = time.perf_counter()
        indices = self._trial_indices(trial_mask)
        return self._fit_objective(
            lambda unit_parameters: self._value_and_grad(unit_parameters, indices),
            lambda candidates: self._screen_values(candidates, indices),
            subject_id,
            started,
        )

    def fit_subjects(self, trial_mask: Bool[Array, " trials"]) -> FitResult:
        """Fit selected subjects independently and combine their results.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Nonempty selection to partition using the dataset's subject labels.

        Returns
        -------
        FitResult
            One result per selected subject, with per-subject diagnostics and
            total elapsed time. The configured budget applies to each fit.
        """
        started = time.perf_counter()
        self._trial_indices(trial_mask)
        masks, subjects = make_subject_trial_masks(
            trial_mask, jnp.asarray(self.subjects)
        )
        results = [
            self.fit(mask, int(subject))
            for mask, subject in zip(masks, subjects, strict=True)
        ]
        combined = results[0].copy()
        combined["fitness"] = [
            value for result in results for value in result["fitness"]
        ]
        combined["nit"] = [value for result in results for value in result["nit"]]
        combined["converged"] = [
            value for result in results for value in result["converged"]
        ]
        combined["fits"] = {
            name: [value for result in results for value in result["fits"][name]]
            for name in results[0]["fits"]
        }
        combined["diagnostics"] = [
            row for result in results for row in result.get("diagnostics", [])
        ]
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def warmup(self, trial_mask: Bool[Array, " trials"]) -> float:
        """Compile the configured evaluation paths for a trial selection.

        Parameters
        ----------
        trial_mask : Bool[Array, " trials"]
            Nonempty selection with one Boolean entry per dataset trial.

        Returns
        -------
        float
            Seconds spent warming loss, gradient and coordinate
            calculations. This does not consume a random restart sequence.
        """
        indices = self._trial_indices(trial_mask)
        started = time.perf_counter()
        self._warmup_objective(
            lambda unit_parameters: self._value_and_grad(unit_parameters, indices),
            lambda candidates: self._screen_values(candidates, indices),
        )
        return time.perf_counter() - started

    def _make_loss_functions(
        self,
    ) -> tuple[
        Callable[[Array, Array], tuple[Array, Array]], Callable[[Array, Array], Array]
    ]:
        """Compile once, accepting selected trial indices as dynamic JAX data."""

        def scalar_loss(
            unit_parameters: Float[Array, " parameters"],
            indices: Integer[Array, " selected_trials"],
        ) -> Float[Array, ""]:
            """Score one candidate in optimization coordinates."""
            parameters = self.parameter_space.to_physical(unit_parameters)
            return self.loss_fn(
                indices, self.base_params, self.names, parameters[:, None]
            )[0]

        @jax.jit
        def screen_values(
            unit_parameters: Float[Array, " samples parameters"],
            indices: Integer[Array, " selected_trials"],
        ) -> Float[Array, " samples"]:
            """Score candidate rows in the separately configured sampling space."""
            parameters = self.sampling_space.to_physical(unit_parameters)
            return self.loss_fn(indices, self.base_params, self.names, parameters.T)

        return jax.jit(jax.value_and_grad(scalar_loss)), screen_values

    def _trial_indices(
        self, trial_mask: Bool[Array, " trials"]
    ) -> Integer[Array, " selected_trials"]:
        """Select dataset rows from a concrete, nonempty Boolean mask."""
        mask = np.asarray(trial_mask)
        return jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)


class ScipyLBFGSB(_ScipyGradient):
    """Fit a differentiable loss with box bounds and reproducible restarts.

    The constructor accepts the same dataset, model factory and loss factory
    as the DE fitters, including configured callable loss constructors.
    Optimization uses explicit unit-interval coordinates for parameter scaling;
    the loss always receives physical parameter values. SciPy owns the local
    optimizer. Only loss evaluation and differentiation use JAX.

    Required hyperparameter: ``bounds`` (a nonempty name -> [lower, upper]
    mapping of finite, strictly ordered bounds). Optional controls:
    ``transforms`` (a name -> 'linear'/'log'/'log1p'/'exp_neg' mapping),
    ``sampling_transforms`` (same syntax, defaults to ``transforms``),
    ``initial_params`` (one complete physical starting mapping, refined first),
    ``best_of`` (maximum local solves, default 8), ``screening_size`` (64),
    ``screening_batch_size`` (16), ``min_start_distance`` (0.1),
    ``seed`` (0), ``num_steps`` (500), ``max_fun`` (15000),
    ``relative_tolerance`` (1e-9), ``gradient_tolerance`` (1e-5),
    ``max_line_search_steps`` (30), ``time_limit_seconds`` (None), and
    ``local_time_limit_seconds`` (None, an allowance for each restart).
    These defaults describe L-BFGS-B; the trust-constr subclass documents its
    solver-specific controls. By default, score a Latin-hypercube bank and refine
    promising, separated candidates. Distances are RMS differences in those
    sampling coordinates; optimization uses ``transforms`` independently.
    Set ``screening_size=0`` for unscreened uniform restarts;
    combine this with ``best_of=1`` and ``initial_params`` for local polishing.
    Transforms are explicit: CMR can use {'primacy_decay': 'exp_neg'} to fit
    retention while preserving its physical decay parameter and bounds.

    Time limits include screening and all restarts and are checked between
    evaluations. One initial value/gradient is always allowed as a fallback,
    recorded explicitly if late. Later improvements are retained only when
    evaluated before both deadlines; in-flight work cannot be interrupted.
    Call ``warmup(mask)`` to measure compilation separately. Without warmup,
    compilation counts toward the time limits.
    Nonfinite values/gradients end that restart, with diagnostics retained;
    a finite objective with an invalid gradient is never reported as a good fit.

    Uses the caller's JAX precision configuration; it never changes global JAX
    settings. Double precision is recommended for tight tolerances and bounds
    near singular model limits. The loss must supply meaningful derivatives;
    differentiability cannot be inferred from the general LossFn protocol.

    Caller requirements: counts are positive integers, except screening_size may be
    zero. min_start_distance lies in [0, 1]; tolerances are finite and nonnegative;
    supplied time limits, trust radii and barrier parameters are finite and positive.
    initial_params must name every free parameter once with finite values inside its
    bounds. L-BFGS-B-only options must not be passed to trust-constr.
    """

    _method = "L-BFGS-B"


class ScipyTrustConstr(_ScipyGradient):
    """Bounded trust-constr with JAX gradients and SciPy's BFGS curvature.

    Accepts the same dataset, model, loss and FitResult contract as ScipyLBFGSB.
    Required option: ``bounds``. Shared controls include ``transforms``,
    ``sampling_transforms``, ``initial_params``, screening, seed, restart count,
    and global/per-restart time limits. Sampling defaults to the optimization
    coordinates; use an explicit mapping to choose a different distribution.

    Defaults: ``best_of=4``, ``screening_size=64``,
    ``screening_batch_size=16``, ``min_start_distance=0.1``,
    ``num_steps=1000``, ``gradient_tolerance=1e-7``,
    ``step_tolerance=1e-12``, ``barrier_tolerance=1e-10``,
    ``initial_trust_radius=1.0``, ``initial_barrier_parameter=0.1``.
    L-BFGS-B's max_fun, relative_tolerance and max_line_search_steps controls
    are not accepted. No Hessian is differentiated or compiled: SciPy updates
    its curvature estimate from the gradients.

    Call warmup(mask) before imposing short fitting budgets. Deadlines include
    screening and restarts and are checked between evaluations. One initial
    evaluation is guaranteed as a fallback; any late fallback is identified in
    diagnostics. Successful solver termination is not proof of a global optimum.
    """

    _method = "trust-constr"
