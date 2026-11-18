"""SciPy differential evolution with JAX population loss evaluation.

SciPy owns search and optional polishing on the host. The loss adapter handles
preparation and coordinate conversion; reported fits always use model units.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import numpy as np
from jax import jit
from jax import numpy as jnp
from scipy.optimize import OptimizeResult, differential_evolution
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
    LossFn,
    MemorySearchModelFactory,
    Parameters,
    RecallDataset,
)

__all__ = ["ScipyDE", "make_scipy_loss_fn"]

type _ScipyLossFn = Callable[
    [
        Float[Array | np.ndarray, " parameters"]
        | Float[Array | np.ndarray, " parameters samples"]
    ],
    Float[Array, ""] | Float[Array, " samples"],
]


def make_scipy_loss_fn(
    loss_fn: LossFn,
    trial_indices: Integer[Array, " trials"],
    base_params: Parameters,
    free_param_names: Iterable[str],
    parameter_space: ParameterSpace | None = None,
) -> _ScipyLossFn:
    """Adapt a population loss to SciPy's scalar and vectorized interfaces.

    Args:
        loss_fn: Ordinary population loss.
        trial_indices: Dataset rows selected for this objective.
        base_params: Fixed parameters and defaults overridden by free values.
        free_param_names: Distinct names in the order of the input parameter axis.
        parameter_space: Optional conversion from unit coordinates to physical
            model values. Its names must exactly match free_param_names in order.
            When omitted, the objective accepts physical values directly.

    Returns:
        A compiled objective accepting shape (parameters,) for one scalar loss
        or (parameters, samples) for one loss per column. SciPy uses the former
        for polishing and the latter for population evaluation. The supplied
        loss always receives physical values shaped (parameters, samples).

    """
    names = tuple(free_param_names)

    def score(
        population: Float[Array, "parameters samples"],
    ) -> Float[Array, " samples"]:
        return loss_fn(trial_indices, base_params, names, population)

    return _make_scipy_objective(score, parameter_space)


def _make_scipy_objective(
    score: Callable[[Float[Array, "parameters samples"]], Float[Array, " samples"]],
    parameter_space: ParameterSpace | None,
) -> _ScipyLossFn:
    """Adapt physical population scoring to SciPy's scalar/vectorized convention."""

    @jit
    def evaluate(
        parameters: Float[Array, " parameters"] | Float[Array, " parameters samples"],
    ) -> Float[Array, ""] | Float[Array, " samples"]:
        # Rank and shape are static under JIT; no traced value chooses a branch.
        population = parameters[:, None] if parameters.ndim == 1 else parameters
        if parameter_space is not None:
            # ParameterSpace uses the final axis, whereas losses use rows.
            population = parameter_space.to_physical(population.T).T
        values = score(population)
        return values[0] if parameters.ndim == 1 else values

    return evaluate


class ScipyDE:
    """Fit bounded parameters with SciPy differential evolution and restarts.

    Bounds, explicit initial populations and returned fits use physical model
    units. Providing ``transforms`` searches in unit coordinates defined by
    ParameterSpace; omitting it preserves the physical-coordinate search.
    Random streams depend on the seed, fit-call index and restart index.
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

    def _configure_optimizer(
        self,
        dataset: RecallDataset,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None,
    ) -> None:
        """Configure bounds, restarts and reporting independently of the objective."""
        options = hyperparams or {}
        self.dataset = dataset
        self.base_params = base_params
        self.subjects = dataset["subject"].flatten()
        self.progress_bar = options.get("progress_bar", True)
        self.display_iterations = options.get("display_iterations", False)
        self.all_hyperparams = _search_options(options)
        self.free_parameter_bounds = self.all_hyperparams["bounds"]
        self.bounds = np.array(list(self.free_parameter_bounds.values()))
        self.parameter_space = (
            ParameterSpace(self.free_parameter_bounds, options["transforms"])
            if options.get("transforms")
            else None
        )
        self._fit_counter = 0

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
        objective = make_scipy_loss_fn(
            self.loss_fn,
            jnp.where(trial_mask)[0],
            self.base_params,
            self.free_parameter_bounds,
            self.parameter_space,
        )
        bounds, initial = self._optimization_domain()
        best = self._best_restart(objective, bounds, initial)
        return self._format_result(best, subject_id, started)

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
        with trange(len(subjects), disable=not self.progress_bar) as progress:
            for index in progress:
                result = self.fit(masks[index], int(subjects[index]))
                if not np.isfinite(result["fitness"][0]):
                    raise ValueError(
                        f"Non-finite fitness for subject {subjects[index]}", result
                    )
                combined["fitness"] += result["fitness"]
                combined["nit"] += result["nit"]
                combined["converged"] += result["converged"]
                combined.setdefault("diagnostics", []).extend(
                    result.get("diagnostics", [])
                )
                for name, values in result["fits"].items():
                    combined["fits"][name] += values
                progress.set_description(
                    f"Subject={subjects[index]}, Fitness={result['fitness'][0]}"
                )
        combined["fit_time"] = time.perf_counter() - started
        return combined

    def _optimization_domain(self) -> tuple[np.ndarray, str | np.ndarray]:
        """Choose search bounds and convert explicit physical starts if needed."""
        initial = self.all_hyperparams["init"]
        if self.parameter_space is None:
            return self.bounds, initial
        bounds = np.tile([0.0, 1.0], (len(self.free_parameter_bounds), 1))
        if not isinstance(initial, str):
            initial = self._initial_population(initial)
        return bounds, initial

    def _initial_population(
        self,
        initial: Float[np.ndarray, " samples parameters"],
    ) -> Float[np.ndarray, " samples parameters"]:
        """Validate physical rows before converting them to unit coordinates."""
        assert self.parameter_space is not None
        values = np.asarray(initial, dtype=float)
        return np.asarray(self.parameter_space.to_unit(jnp.asarray(values)))

    def _best_restart(
        self,
        objective: _ScipyLossFn,
        bounds: Float[np.ndarray, " parameters 2"],
        initial: str | np.ndarray,
    ) -> OptimizeResult:
        """Run independent random streams and retain the strict best result."""
        fit_counter = self._fit_counter
        self._fit_counter += 1
        best = None
        best_fitness = np.inf
        restart_history = []
        for index in range(self.all_hyperparams["best_of"]):
            rng = np.random.default_rng(
                [self.all_hyperparams["seed"], fit_counter, index]
            )
            result = self._run_restart(objective, bounds, initial, rng)
            physical = (
                result.x
                if self.parameter_space is None
                else np.asarray(self.parameter_space.to_physical(jnp.asarray(result.x)))
            )
            restart_history.append(
                {
                    "restart": index,
                    "seed_components": [
                        self.all_hyperparams["seed"],
                        fit_counter,
                        index,
                    ],
                    "fitness": float(result.fun) if np.isfinite(result.fun) else None,
                    "nit": int(result.nit),
                    "converged": bool(result.success),
                    "message": str(getattr(result, "message", "")),
                    "parameters": dict(
                        zip(
                            self.free_parameter_bounds,
                            map(float, physical),
                            strict=True,
                        )
                    )
                    if np.isfinite(result.fun)
                    else None,
                }
            )
            if result.fun < best_fitness:
                best, best_fitness = result, result.fun
                best.selected_restart = index
        assert best is not None, "No fit result found"
        best.restart_history = restart_history
        return best

    def _run_restart(
        self,
        objective: _ScipyLossFn,
        bounds: Float[np.ndarray, " parameters 2"],
        initial: str | np.ndarray,
        rng: np.random.Generator,
    ) -> OptimizeResult:
        """Translate library hyperparameters to one vectorized SciPy DE call."""
        options = self.all_hyperparams
        return differential_evolution(
            objective,
            bounds,
            maxiter=options["num_steps"],
            popsize=options["pop_size"],
            vectorized=True,
            disp=self.display_iterations,
            tol=options["relative_tolerance"],
            atol=options["absolute_tolerance"],
            mutation=options["diff_w"],
            recombination=options["cross_over_rate"],
            # SciPy accepts arrays here, despite its narrower bundled annotation.
            init=cast(Any, initial),
            polish=options["polish"],
            rng=rng,
        )

    def _empty_result(self) -> FitResult:
        """Create shared metadata with empty, aligned per-subject columns."""
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
        self, best: OptimizeResult, subject_id: int, started: float
    ) -> FitResult:
        """Report one restart in physical units, with fitted values overriding defaults."""
        physical = best.x
        if self.parameter_space is not None:
            physical = np.asarray(
                self.parameter_space.to_physical(jnp.asarray(physical))
            )
        parameters = dict(self.base_params) | dict(
            zip(self.free_parameter_bounds, physical, strict=True)
        )
        result = self._empty_result()
        result["fitness"] = [float(best.fun)]
        result["nit"] = [int(best.nit)]
        result["converged"] = [bool(best.success)]
        result["fits"] = {
            name: [np.asarray(value).item()] for name, value in parameters.items()
        }
        result["fits"]["subject"] = [subject_id]
        result["fit_time"] = time.perf_counter() - started
        result["diagnostics"] = [
            {
                "subject": subject_id,
                "selected_restart": best.selected_restart,
                "restarts": best.restart_history,
            }
        ]
        return result


def _search_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize DE defaults and scalar mutation settings without changing search units."""
    diff_w = options.get("diff_w", (0.5, 1.0))
    if not isinstance(diff_w, (tuple, list)):
        diff_w = (diff_w, diff_w)
    return {
        "bounds": options.get("bounds", {}),
        "num_steps": options.get("num_steps", 1000),
        "pop_size": options.get("pop_size", 15),
        "relative_tolerance": options.get("relative_tolerance", 0.001),
        "absolute_tolerance": options.get("absolute_tolerance", 0.0),
        "cross_over_rate": options.get("cross_over_rate", 0.9),
        "diff_w": tuple(diff_w),
        "best_of": options.get("best_of", 1),
        "init": options.get("init", "latinhypercube"),
        "polish": options.get("polish", True),
        "seed": int(options.get("seed", 0)),
        "transforms": dict(options.get("transforms", {})),
    }
