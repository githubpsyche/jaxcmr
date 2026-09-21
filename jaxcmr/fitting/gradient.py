"""Bounded SciPy fitting using compiled JAX values and gradients.

Each fit prepares its loss, screens candidate starts, runs independent local
restarts, and reports the best valid evaluation. SciPy and deadline accounting
run on the host; parameter conversion, loss evaluation and gradients use JAX.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import BFGS, Bounds, OptimizeResult, minimize
from scipy.stats import qmc

from jaxcmr.fitting._preparation import LossPreparation
from jaxcmr.fitting.initialization import select_starts
from jaxcmr.fitting.parameters import ParameterSpace
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
    RecallDataset,
)

__all__ = ["ScipyLBFGSB", "ScipyTrustConstr"]

type _ValueAndGradientFn = Callable[[Array, PreparedLoss[Any]], tuple[Array, Array]]
type _ScreenValuesFn = Callable[[Array, PreparedLoss[Any]], Array]
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
        self, value: float, unit: np.ndarray, restart: int, observed: float
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
                "value": value,
                "unit": self.best_unit.tolist(),
                "restart": restart,
            }
        )


class _ScipyGradient:
    """Fit a differentiable loss with box bounds and reproducible restarts.

    The model/loss constructor and returned FitResult match the DE fitters.
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
    """

    _method: str

    def __init__(
        self,
        dataset: RecallDataset,
        features: Float[Array, "word_pool_items features_count"] | None,
        base_params: Mapping[str, Float_],
        model_factory_cls: type[MemorySearchModelFactory],
        loss_fn_cls: type[LossFn],
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Configure parameter coordinates, solver controls and compiled losses.

        Parameters
        ----------
        dataset : RecallDataset
            Observations with one subject label per trial.
        features : Array or None
            Optional vocabulary features passed to the loss constructor.
        base_params : Mapping[str, Float_]
            Model defaults and fixed values; free parameters override defaults.
        model_factory_cls : type[MemorySearchModelFactory]
            Factory passed to the loss constructor.
        loss_fn_cls : type[LossFn]
            Differentiable population loss, optionally supporting preparation.
        hyperparams : dict[str, Any] or None
            Bounds and fitting controls documented on this class and its solver.
        """
        options = dict(hyperparams or {})
        self.dataset = dataset
        self.subjects = np.asarray(dataset["subject"]).reshape(-1)
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
        self._validate_hyperparameters()
        self._initial = self._initial_parameters()
        self._fit_counter = 0
        self.loss_fn = loss_fn_cls(model_factory_cls, dataset, features)
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
        self._losses = LossPreparation(self.loss_fn)
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
            per-restart diagnostics. Elapsed time includes preparation,
            compilation, screening and every restart unless warmed beforehand.
        """
        started = time.perf_counter()
        indices = self._trial_indices(trial_mask)
        prepared = self._losses.select(indices)
        config = self.all_hyperparams
        rng = np.random.default_rng([int(config["seed"]), self._fit_counter])
        self._fit_counter += 1
        limit = config["time_limit_seconds"]
        progress = _FitProgress(started, np.inf if limit is None else started + limit)
        units, selected, screening = self._starting_points(prepared, rng, progress)
        for restart, candidate_index in enumerate(selected):
            if progress.records and time.perf_counter() >= progress.deadline:
                break
            record = self._run_restart(
                prepared,
                units[candidate_index],
                restart,
                int(candidate_index),
                progress,
            )
            progress.records.append(record)
        return self._format_result(progress, screening, subject_id)

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
            if np.any(mask)
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
        combined["diagnostics"] = {
            "subjects": [result.get("diagnostics", {}) for result in results]
        }
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
            Seconds spent preparing and warming loss, gradient and coordinate
            calculations. This does not consume a random restart sequence.
        """
        indices = self._trial_indices(trial_mask)
        started = time.perf_counter()
        prepared = self._losses.select(indices)
        point = jnp.full(len(self.names), 0.5, dtype=self._lower.dtype)
        jax.block_until_ready(self._value_and_grad(point, prepared))
        self._physical_parameters(point).block_until_ready()
        self._unit_parameters(self._physical_parameters(point)).block_until_ready()
        self._sampling_parameters(self._physical_parameters(point)).block_until_ready()
        count = (
            self.all_hyperparams["screening_size"] or self.all_hyperparams["best_of"]
        )
        self._sampling_to_unit(jnp.tile(point, (count, 1))).block_until_ready()
        if self.all_hyperparams["screening_size"]:
            batch = jnp.tile(point, (self.all_hyperparams["screening_batch_size"], 1))
            self._screen_values(batch, prepared).block_until_ready()
        return time.perf_counter() - started

    def prepare_selections(
        self, trial_masks: Iterable[Bool[Array, " trials"]]
    ) -> float:
        """Prepare compatible observation layouts for subsequent fitting calls.

        Parameters
        ----------
        trial_masks : Iterable[Bool[Array, " trials"]]
            Anticipated selections. Requires optional batch preparation on the
            loss. A successful call replaces the plan; empty input clears it.

        Returns
        -------
        float
            Preparation time in seconds, without compilation or fitting. Planned
            data remain available until replaced; unplanned selections are
            prepared individually. Compatible arrays can share compilation.
        """
        started = time.perf_counter()
        self._losses.prepare_selections(trial_masks, self.subjects.shape)
        return time.perf_counter() - started

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
                if key in options:
                    raise ValueError(f"{key} applies to L-BFGS-B, not trust-constr.")
                del config[key]
            config.update(
                step_tolerance=options.get("step_tolerance", 1e-12),
                barrier_tolerance=options.get("barrier_tolerance", 1e-10),
                initial_trust_radius=options.get("initial_trust_radius", 1.0),
                initial_barrier_parameter=options.get("initial_barrier_parameter", 0.1),
            )
        return config

    def _validate_hyperparameters(self) -> None:
        """Reject invalid restart counts, tolerances and time/solver allowances."""
        for name in (
            "best_of",
            "num_steps",
            "max_fun",
            "max_line_search_steps",
            "screening_batch_size",
        ):
            if name not in self.all_hyperparams:
                continue
            value = self.all_hyperparams[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        size = self.all_hyperparams["screening_size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("screening_size must be a nonnegative integer.")
        distance = self.all_hyperparams["min_start_distance"]
        if not np.isfinite(distance) or not 0 <= distance <= 1:
            raise ValueError("min_start_distance must lie between zero and one.")
        for name in (
            "relative_tolerance",
            "gradient_tolerance",
            "step_tolerance",
            "barrier_tolerance",
        ):
            if name not in self.all_hyperparams:
                continue
            value = self.all_hyperparams[name]
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "time_limit_seconds",
            "local_time_limit_seconds",
            "initial_trust_radius",
            "initial_barrier_parameter",
        ):
            limit = self.all_hyperparams.get(name)
            if limit is not None and (not np.isfinite(limit) or limit <= 0):
                raise ValueError(f"{name} must be positive and finite.")

    def _initial_parameters(self) -> np.ndarray | None:
        """Validate an optional complete starting vector in physical parameter units."""
        initial = self.all_hyperparams["initial_params"]
        if initial is not None:
            if not isinstance(initial, Mapping) or set(initial) != set(self.names):
                raise ValueError(
                    "initial_params must name every free parameter exactly once."
                )
            initial = np.array([initial[name] for name in self.names], dtype=float)
            if (
                not np.all(np.isfinite(initial))
                or np.any(initial < self.bounds[:, 0])
                or np.any(initial > self.bounds[:, 1])
            ):
                raise ValueError("initial_params must be finite and within bounds.")
        return initial

    def _make_loss_functions(self) -> tuple[_ValueAndGradientFn, _ScreenValuesFn]:
        """Compile one gradient and screening path for every supported loss route.

        LossPreparation binds observations before these functions are called.
        Explicit data remain dynamic JAX arguments, so compatible selections
        reuse compilation without either function capturing a current subject.
        """

        def scalar_loss(unit_parameters: Array, prepared: PreparedLoss[Any]) -> Array:
            """Score one candidate in optimization coordinates."""
            parameters = self.parameter_space.to_physical(unit_parameters)
            return prepared(self.base_params, self.names, parameters[:, None])[0]

        @jax.jit
        def screen_values(unit_parameters: Array, prepared: PreparedLoss[Any]) -> Array:
            """Score candidate rows in the separately configured sampling space."""
            parameters = self.sampling_space.to_physical(unit_parameters)
            return prepared(self.base_params, self.names, parameters.T)

        return jax.jit(jax.value_and_grad(scalar_loss)), screen_values

    def _trial_indices(self, trial_mask: Array) -> Array:
        """Validate one concrete nonempty Boolean mask and select its dataset rows."""
        mask = np.asarray(trial_mask)
        if (
            mask.shape != self.subjects.shape
            or mask.dtype != np.bool_
            or not np.any(mask)
        ):
            raise ValueError(
                "trial_mask must be a nonempty Boolean selection with one entry per trial."
            )
        return jnp.asarray(np.flatnonzero(mask), dtype=jnp.int32)

    def _starting_points(
        self,
        prepared: PreparedLoss[Any],
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
                candidates, prepared, progress.deadline
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
        self, candidates: np.ndarray, prepared: PreparedLoss[Any], deadline: float
    ) -> tuple[np.ndarray, int]:
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
                self._screen_values(
                    jnp.asarray(padded, dtype=self._lower.dtype), prepared
                )
            )[: len(batch)]
            evaluated += len(batch)
        return scores, evaluated

    def _run_restart(
        self,
        prepared: PreparedLoss[Any],
        initial: np.ndarray,
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
        }
        started = time.perf_counter()
        local_limit = self.all_hyperparams["local_time_limit_seconds"]
        deadline = (
            min(progress.deadline, started + local_limit)
            if local_limit is not None
            else progress.deadline
        )
        evaluate = self._restart_objective(prepared, record, progress, deadline)

        def callback(_unit: np.ndarray, _state: OptimizeResult | None = None) -> None:
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
        prepared: PreparedLoss[Any],
        record: dict[str, Any],
        progress: _FitProgress,
        deadline: float,
    ) -> _ScipyObjective:
        """Bind one restart's objective, enforcing deadlines around each evaluation.

        The first fit evaluation is a fallback even if it finishes late. Later
        evaluations must finish within both budgets to update the reported best.
        Invalid gradients reject a candidate even when its loss is finite.
        """

        def evaluate(unit: np.ndarray) -> tuple[float, np.ndarray]:
            initial_fallback = record["nfev"] == 0 and record["restart"] == 0
            if not initial_fallback and time.perf_counter() >= deadline:
                raise _BudgetExpired
            value, gradient = self._value_and_grad(
                jnp.asarray(unit, dtype=self._lower.dtype), prepared
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
                progress.consider(value, unit, record["restart"], observed)
            return value, gradient

        return evaluate

    def _minimize(
        self,
        evaluate: _ScipyObjective,
        initial: np.ndarray,
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
            bounds=Bounds(
                np.zeros(len(self.names)), np.ones(len(self.names)), keep_feasible=True
            ),
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
            converged=bool(result.success),
            message=str(result.message),
            final_value=float(result.fun),
            final_unit=result.x.tolist(),
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
            "fixed": {key: float(value) for key, value in self.base_params.items()},
            "free": self.free_parameter_bounds,
            "fitness": [progress.best_value],
            "nit": [chosen["nit"]],
            "converged": [bool(converged)],
            "fits": {
                **{key: [float(value)] for key, value in self.base_params.items()},
                **{key: [float(params[i])] for i, key in enumerate(self.names)},
                "subject": [subject_id],
            },
            "hyperparameters": dict(self.all_hyperparams),
            "fit_time": time.perf_counter() - progress.started,
            "diagnostics": {
                "restarts": progress.records,
                "best_restart": progress.best_restart,
                "screening": screening,
                "history": progress.history,
                "nfev": sum(record["nfev"] for record in progress.records),
                "jax_dtype": str(self._lower.dtype),
                "prepared_trial_count": self._losses.prepared_trial_count,
                "effective_bounds": np.stack(
                    [np.asarray(self._lower), np.asarray(self._upper)], axis=1
                ).tolist(),
            },
        }


class ScipyLBFGSB(_ScipyGradient):
    """Bounded L-BFGS-B with JAX derivatives, screening and restarts.

    Accepts the shared gradient-fitting controls documented on
    ``_ScipyGradient``. Existing defaults and physical parameter conventions
    are unchanged; sampling transforms and local time limits are opt-in.
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
