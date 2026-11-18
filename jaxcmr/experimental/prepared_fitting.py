"""Fit a differentiable loss to observations the caller has already selected.

For participant fits, select one participant's trials and prepare a
``PreparedLoss`` that scores candidate parameters against those trials.
Pass it to ``PreparedScipyLBFGSB.fit`` or ``PreparedScipyTrustConstr.fit``
with the participant ID. Each call returns one fit row for those observations.

A fitter can be used again with the next participant's prepared loss.
``warmup`` compiles the evaluator before fitting when preparation and fit
time need to be measured separately. Compatible observations and the same
evaluation function can share that compilation. Parameter bounds, restarts
and result rows follow the conventions of the dataset-based gradient fitters.
"""

import time
from typing import Any

import jax

from jaxcmr.experimental.prepared import PreparedLoss
from jaxcmr.fitting.gradient import _GradientOptimizer
from jaxcmr.typing import Array, FitResult, Float, Parameters

__all__ = ["PreparedScipyLBFGSB", "PreparedScipyTrustConstr"]


class _PreparedGradient(_GradientOptimizer):
    """Bind dynamic prepared observations to the existing optimizer."""

    def __init__(
        self, base_params: Parameters, hyperparams: dict[str, Any] | None = None
    ) -> None:
        self._configure_optimizer(base_params, hyperparams)

        def scalar_loss(
            unit: Float[Array, " parameters"], loss: PreparedLoss
        ) -> Float[Array, ""]:
            physical = self.parameter_space.to_physical(unit)
            return loss(self.base_params, self.names, physical[:, None])[0]

        @jax.jit
        def screen_values(
            units: Float[Array, "candidates parameters"], loss: PreparedLoss
        ) -> Float[Array, " candidates"]:
            physical = self.sampling_space.to_physical(units)
            return loss(self.base_params, self.names, physical.T)

        self._value_and_grad = jax.jit(jax.value_and_grad(scalar_loss))
        self._screen_values = screen_values

    def warmup(self, prepared_loss: PreparedLoss) -> float:
        """Compile evaluation paths without consuming a restart sequence.

        Parameters
        ----------
        prepared_loss : PreparedLoss
            Observations and a stable, differentiable population evaluator.

        Returns
        -------
        float
            Elapsed compilation and warmup time in seconds.
        """
        started = time.perf_counter()
        self._warmup_objective(
            lambda unit: self._value_and_grad(unit, prepared_loss),
            lambda units: self._screen_values(units, prepared_loss),
        )
        return time.perf_counter() - started

    def fit(self, prepared_loss: PreparedLoss, subject_id: int = -1) -> FitResult:
        """Fit supplied observations and retain the best usable evaluation.

        Parameters
        ----------
        prepared_loss : PreparedLoss
            Explicitly prepared observations and their population scorer.
        subject_id : int, optional
            Label for the fitted row; -1 denotes a pooled fit by convention.

        Returns
        -------
        FitResult
            Physical parameters and objective, with one diagnostics row recording
            screening, every attempted restart and deadline information. Without
            warmup, compilation counts toward the configured time allowance.
        """
        started = time.perf_counter()
        return self._fit_objective(
            lambda unit: self._value_and_grad(unit, prepared_loss),
            lambda units: self._screen_values(units, prepared_loss),
            subject_id,
            started,
        )


class PreparedScipyLBFGSB(_PreparedGradient):
    """Fit a PreparedLoss using bounded L-BFGS-B and JAX derivatives.

    Parameters
    ----------
    base_params : Parameters
        Fixed model values and defaults overridden by free parameters.
    hyperparams : dict or None, optional
        The same bounds, transforms, screening, restart and deadline controls as
        ScipyLBFGSB. Preparation and participant iteration belong to the caller.

    Notes
    -----
    Call warmup(prepared_loss) to separate compilation from fitting time, then
    fit(prepared_loss, subject_id=participant). Compatible arrays can change
    between calls without recreating the evaluator or this fitter.
    """

    _method = "L-BFGS-B"


class PreparedScipyTrustConstr(_PreparedGradient):
    """Fit a PreparedLoss using trust-constr and JAX derivatives.

    Parameters
    ----------
    base_params : Parameters
        Fixed model values and defaults overridden by free parameters.
    hyperparams : dict or None, optional
        The same bounds, transforms, screening, restart and deadline controls as
        ScipyTrustConstr, including its trust-radius and barrier settings.

    Notes
    -----
    Call warmup(prepared_loss) before imposing a short time budget. Pass prepared
    observations explicitly to each fit; no model or selection discovery occurs.
    """

    _method = "trust-constr"
