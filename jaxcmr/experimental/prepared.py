"""Evaluate candidate parameters against observations prepared by a model.

For a model that predicts individual recall events, ``PreparedPrediction``
holds the observations, the function that predicts their probabilities,
and masks identifying which events to score. ``prediction_likelihood``
combines those probabilities into one negative log likelihood per candidate.
The model supplies probabilities conditioned on the full response history.

``PreparedLoss`` pairs prepared observations with a function that directly
scores a set of candidates. It can use the likelihood calculation above
or a model-specific objective. Keeping observation arrays as arguments
lets JAX reuse a compiled scoring function when the next participant's
data have compatible shapes and types.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxcmr.typing import Array, Bool, Float, Parameters

type PredictionFn[DataT] = Callable[[DataT, Parameters], tuple[Array, ...]]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedPrediction[DataT]:
    """Immutable prediction plan with dynamic observations and static code.

    ``evaluate`` accepts data and a complete model parameter mapping and returns
    probability blocks. ``masks`` contains the caller's scoring requests in the
    same block layout. Reordering or padding must preserve observation order
    and multiplicity. The caller must snapshot mutable source data during preparation.

    Attributes
    ----------
    data : DataT
        Model-specific observation arrays in a JAX-compatible pytree. These
        remain dynamic arguments, permitting compilation reuse across datasets.
    masks : tuple[Array, ...]
        Boolean scoring requests, with one mask per returned probability block.
    evaluate : PredictionFn[DataT]
        Pure model calculation, stored as static JIT metadata. Keep observations
        in data rather than its closure when compilation should be reusable.

    """

    data: DataT
    masks: tuple[Bool[Array, "..."], ...]
    evaluate: PredictionFn[DataT] = field(metadata={"static": True})

    def probabilities(self, parameters: Parameters) -> tuple[Array, ...]:
        """Predict requested events, retaining the full conditioning history.

        Parameters
        ----------
        parameters : Parameters
            Complete model parameter mapping, including fixed configuration.

        Returns
        -------
        tuple[Array, ...]
            Probability blocks with the same shapes and ordering as masks.
            Unrequested entries may contain fillers; consumers must mask them
            before a transformation such as log that excludes those fillers.

        """
        return self.evaluate(self.data, parameters)


type PreparedEvaluationFn[DataT] = Callable[
    [DataT, Parameters, Iterable[str], Float[Array, "parameters n_samples"]],
    Float[Array, " n_samples"],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedLoss[DataT]:
    """A population loss with explicit data and a stable evaluation function.

    ``data`` must be a JAX-compatible pytree. ``evaluate`` accepts data, base
    parameters, free parameter names and a parameter matrix, returning one
    loss per column. Use a stable, pure function; observations belong in data,
    not in its closure. The function is static JIT metadata. Data leaves remain
    dynamic, so compatible shapes can share compilation across selections.

    Preparation must snapshot mutable source data. Freezing this container
    prevents attribute reassignment but cannot freeze arbitrary Python objects
    supplied by a caller. JAX arrays and immutable containers are recommended.
    Use this object with the explicit experimental prepared fitters.

    Attributes
    ----------
    data : DataT
        Prepared observations and masks, represented as a JAX-compatible pytree.
    evaluate : PreparedEvaluationFn[DataT]
        Pure evaluation function, stored as static metadata. Its output is one
        loss per parameter column, matching the existing fitter convention.

    """

    data: DataT
    evaluate: PreparedEvaluationFn[DataT] = field(metadata={"static": True})

    def __call__(
        self,
        base: Parameters,
        names: Iterable[str],
        parameters: Float[Array, "parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        """Evaluate a population of parameter vectors against prepared data.

        Parameters
        ----------
        base : Parameters
            Fixed parameters and defaults for the model.
        names : Iterable[str]
            Free parameter names in the same order as parameter rows.
        parameters : Float[Array, "parameters n_samples"]
            One candidate parameter vector per column; overrides base values.

        Returns
        -------
        Float[Array, "n_samples"]
            Loss for each candidate, in column order.

        """
        return self.evaluate(self.data, base, names, parameters)


def prediction_likelihood(
    prediction: PreparedPrediction,
    base: Parameters,
    names: Iterable[str],
    parameters: Float[Array, "parameters n_samples"],
) -> Float[Array, " n_samples"]:
    """Generic masked NLL reduction; all response probabilities come from a model.

    Mask probabilities before taking logs, including impossible unscored STOP
    events. A model must have already conditioned on masked real responses.

    Parameters
    ----------
    prediction : PreparedPrediction
        Model-owned probability blocks and their aligned scoring masks.
    base : Parameters
        Complete defaults, including model configuration flags.
    names : Iterable[str]
        Distinct free parameter names, ordered like the rows of parameters.
    parameters : Float[Array, "parameters n_samples"]
        Parameter population, with one candidate per column.

    Returns
    -------
    Float[Array, "n_samples"]
        Summed negative log-likelihood for each candidate. Impossible scored
        events yield infinity; masked events make no contribution.

    """
    names = tuple(names)

    def score(column):
        model_parameters = dict(base) | dict(zip(names, column, strict=True))
        total = jnp.asarray(0.0, dtype=parameters.dtype)
        for probabilities, mask in zip(
            prediction.probabilities(model_parameters), prediction.masks, strict=True
        ):
            total -= jnp.log(jnp.where(mask, probabilities, 1.0)).sum()
        return total

    return jax.vmap(score, in_axes=1)(parameters)


__all__ = ["PreparedLoss", "PreparedPrediction", "prediction_likelihood"]
