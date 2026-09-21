"""Prepared observations that can be dynamic arguments to a compiled loss."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxcmr.models.sequence import PreparedPrediction
from jaxcmr.typing import Array, Float, Float_

__all__ = ["PreparedLoss"]

type PreparedEvaluationFn[DataT] = Callable[
    [DataT, Mapping[str, Float_], Iterable[str], Float[Array, "parameters n_samples"]],
    Float[Array, " n_samples"],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedLoss[DataT]:
    """A PreparedLossFn with explicit data and a stable evaluation function.

    ``data`` must be a JAX-compatible pytree. ``evaluate`` accepts data, base
    parameters, free parameter names and a parameter matrix, returning one
    loss per column. Use a stable, pure function; observations belong in data,
    not in its closure. The function is static JIT metadata. Data leaves remain
    dynamic, so compatible shapes can share compilation across selections.

    Preparation must snapshot mutable source data. Freezing this container
    prevents attribute reassignment but cannot freeze arbitrary Python objects
    supplied by a caller. JAX arrays and immutable containers are recommended.
    This callable also works with consumers of ordinary PreparedLossFn objects.

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
        base: Mapping[str, Float_],
        names: Iterable[str],
        parameters: Float[Array, "parameters n_samples"],
    ) -> Float[Array, " n_samples"]:
        """Evaluate a population of parameter vectors against prepared data.

        Parameters
        ----------
        base : Mapping[str, Float_]
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
    base: Mapping[str, Float_],
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
    base : Mapping[str, Float_]
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
    if (
        len(set(names)) != len(names)
        or parameters.ndim != 2
        or parameters.shape[0] != len(names)
    ):
        raise ValueError(
            "Provide distinct free names and a (parameters, samples) array."
        )

    def score(column):
        model_parameters = dict(base) | dict(zip(names, column, strict=True))
        total = jnp.asarray(0.0, dtype=parameters.dtype)
        for probabilities, mask in zip(
            prediction.probabilities(model_parameters), prediction.masks, strict=True
        ):
            if probabilities.shape != mask.shape:
                raise ValueError(
                    "Model probability blocks must match their scoring masks."
                )
            total -= jnp.log(jnp.where(mask, probabilities, 1.0)).sum()
        return total

    return jax.vmap(score, in_axes=1)(parameters)
