"""Explicit experimental fitting for OptimizedCMR's prepared sequence evaluator.

These fitters accept a dataset, fixed parameters and the ordinary optimizer
options. The model is always OptimizedCMR and has no semantic feature pathway.
By default, likelihood conditions on observed recall length (STOP is unscored).
Supply mask_likelihoods to change which observed events contribute to the loss.

prepare_selections(masks) is optional: it prepares compatible participant
layouts together, allowing them to reuse compiled code. Unplanned selections
remain usable. Ordinary fitters do not import or discover this module.
"""

import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxcmr.experimental.optimized_cmr import (
    _CMRSelections,
    _PreparedSequence,
    _sequence_likelihood,
)
from jaxcmr.fitting.evosax import (
    EvosaxDE as _OrdinaryEvosaxDE,
)
from jaxcmr.fitting.evosax import (
    _RestartResult,
    _RunArrays,
)
from jaxcmr.fitting.gradient import _ScipyGradient
from jaxcmr.fitting.scipy import ScipyDE as _OrdinaryScipyDE
from jaxcmr.fitting.scipy import _make_scipy_objective
from jaxcmr.loss.sequence_likelihood import mask_trailing_terminations
from jaxcmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    LikelihoodMaskFn,
    Parameters,
    PRNGKeyArray,
    RecallDataset,
)

__all__ = ["EvosaxDE", "ScipyDE", "ScipyLBFGSB", "ScipyTrustConstr"]


class _CMRFitting:
    """Selection preparation shared only by the experimental CMR fitters."""

    _sequences: _CMRSelections

    def prepare_selections(
        self, trial_masks: Iterable[Bool[Array, " trials"]]
    ) -> float:
        """Prepare layouts without compiling; return elapsed seconds.

        Duplicate selections are prepared once. Empty input clears the plan;
        each supplied mask must select observations. Failure preserves the
        previous plan, and unplanned selections can still be fitted.
        """
        started = time.perf_counter()
        self._sequences.prepare_selections(trial_masks)
        return time.perf_counter() - started

    def _select(self, trial_mask: Bool[Array, " trials"]) -> _PreparedSequence:
        """Resolve a nonempty fit selection before entering compiled code."""
        return self._sequences.select(
            jnp.asarray(np.flatnonzero(trial_mask), dtype=jnp.int32)
        )


class _GradientCMR(_CMRFitting, _ScipyGradient):
    """Bind prepared CMR data to the shared screening and local-search routines."""

    def __init__(
        self,
        dataset: RecallDataset,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None = None,
        *,
        mask_likelihoods: LikelihoodMaskFn = mask_trailing_terminations,
    ) -> None:
        """Configure the ordinary solver controls and experimental CMR scorer."""
        self.dataset = dataset
        self.subjects = np.asarray(dataset["subject"]).reshape(-1)
        self._configure_optimizer(base_params, hyperparams)
        self._sequences = _CMRSelections(dataset, mask_likelihoods)
        self._prepared_value_and_grad, self._prepared_screen_values = (
            self._compile_likelihood()
        )

    def _compile_likelihood(
        self,
    ) -> tuple[
        Callable[[Array, _PreparedSequence], tuple[Array, Array]],
        Callable[[Array, _PreparedSequence], Array],
    ]:
        """Keep compiled functions stable while passing prepared arrays dynamically."""

        def scalar_loss(
            unit_parameters: Float[Array, " parameters"],
            sequence: _PreparedSequence,
        ) -> Float[Array, ""]:
            parameters = self.parameter_space.to_physical(unit_parameters)
            return _sequence_likelihood(
                sequence, self.base_params, self.names, parameters[:, None]
            )[0]

        @jax.jit
        def screen_values(
            unit_parameters: Float[Array, "samples parameters"],
            sequence: _PreparedSequence,
        ) -> Float[Array, " samples"]:
            parameters = self.sampling_space.to_physical(unit_parameters)
            return _sequence_likelihood(
                sequence, self.base_params, self.names, parameters.T
            )

        return jax.jit(jax.value_and_grad(scalar_loss)), screen_values

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit prepared observations using the unchanged screening/restart rules."""
        started = time.perf_counter()
        sequence = self._select(trial_mask)
        result = self._fit_objective(
            lambda unit_parameters: self._prepared_value_and_grad(
                unit_parameters, sequence
            ),
            lambda candidates: self._prepared_screen_values(candidates, sequence),
            subject_id,
            started,
        )
        result.get("diagnostics", [])[0]["prepared_trial_count"] = int(
            np.count_nonzero(trial_mask)
        )
        return result

    def warmup(self, trial_mask: Bool[Array, " trials"]) -> float:
        """Prepare and compile a selection without advancing the restart sequence."""
        started = time.perf_counter()
        sequence = self._select(trial_mask)
        self._warmup_objective(
            lambda unit_parameters: self._prepared_value_and_grad(
                unit_parameters, sequence
            ),
            lambda candidates: self._prepared_screen_values(candidates, sequence),
        )
        return time.perf_counter() - started


class ScipyLBFGSB(_GradientCMR):
    """Experimental prepared CMR with the ordinary L-BFGS-B controls and results."""

    _method = "L-BFGS-B"


class ScipyTrustConstr(_GradientCMR):
    """Experimental prepared CMR with the ordinary trust-constr controls and results."""

    _method = "trust-constr"


class EvosaxDE(_CMRFitting, _OrdinaryEvosaxDE):
    """Experimental prepared CMR using the shared serial evolutionary optimizer."""

    def __init__(
        self,
        dataset: RecallDataset,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None = None,
        *,
        mask_likelihoods: LikelihoodMaskFn = mask_trailing_terminations,
    ) -> None:
        """Configure evolution and a stable compiled prepared-sequence runner."""
        self._configure_optimizer(dataset, base_params, hyperparams)
        self._sequences = _CMRSelections(dataset, mask_likelihoods)
        self._compiled_prepared_run = jax.jit(self._run_prepared)

    def _run_prepared(
        self, key: PRNGKeyArray, sequence: _PreparedSequence
    ) -> _RunArrays:
        """Bind CMR scoring to the same DE calculation used by ordinary fitting."""

        def evaluate(
            population: Float[Array, "candidates parameters"],
        ) -> Float[Array, " candidates"]:
            values = _sequence_likelihood(
                sequence,
                self.base_params,
                self.free_parameter_bounds,
                population.T,
            )
            return self._sanitize_fitness(values, population.shape[0])

        return self._run_generations(key, evaluate)

    def _run_prepared_once(
        self, key: PRNGKeyArray, sequence: _PreparedSequence
    ) -> _RestartResult:
        """Transfer a completed compiled restart to host result values."""
        fitness, parameters, generations, converged = self._compiled_prepared_run(
            key, sequence
        )
        return _RestartResult(
            float(fitness),
            np.asarray(parameters, dtype=float),
            int(generations),
            bool(converged),
        )

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Fit independent restarts and retain the best result."""
        started = time.perf_counter()
        sequence = self._select(trial_mask)
        result = self._run_restarts(
            self._next_key(),
            partial(self._run_prepared_once, sequence=sequence),
        )
        return self._format_result(result, subject_id, started)

    def fit_once(
        self,
        trial_mask: Bool[Array, " trials"],
        key: PRNGKeyArray,
        subject_id: int = -1,
    ) -> FitResult:
        """Run one supplied-key restart without advancing fit's key sequence."""
        started = time.perf_counter()
        sequence = self._select(trial_mask)
        result = self._run_prepared_once(key, sequence)
        return self._format_result([result], subject_id, started, best_of=1)


class ScipyDE(_CMRFitting, _OrdinaryScipyDE):
    """Experimental prepared CMR using the ordinary SciPy DE search controls."""

    def __init__(
        self,
        dataset: RecallDataset,
        base_params: Parameters,
        hyperparams: dict[str, Any] | None = None,
        *,
        mask_likelihoods: LikelihoodMaskFn = mask_trailing_terminations,
    ) -> None:
        """Configure the search domain and CMR observation preparation."""
        self._configure_optimizer(dataset, base_params, hyperparams)
        self._sequences = _CMRSelections(dataset, mask_likelihoods)

    def fit(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Prepare a selection, then run the shared SciPy objective/restart adapter."""
        started = time.perf_counter()
        sequence = self._select(trial_mask)
        names = tuple(self.free_parameter_bounds)
        objective = _make_scipy_objective(
            partial(_sequence_likelihood, sequence, self.base_params, names),
            self.parameter_space,
        )
        bounds, initial = self._optimization_domain()
        best = self._best_restart(objective, bounds, initial)
        return self._format_result(best, subject_id, started)
