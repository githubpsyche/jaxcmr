"""Termination policies for memory search models.

Provides functions that compute the probability of terminating
recall at each retrieval step, including fixed-rate and
activation-dependent termination rules.

Probability calculations retain their floating dtype, including explicit
float32 inputs when JAX enables float64. The numerical margin ``FLOAT32_EPS`` keeps
its historical value; it does not choose the precision of the calculation.

"""

from __future__ import annotations

from typing import Protocol

from jax import eval_shape, lax
from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.math import FLOAT32_EPS
from jaxcmr.typing import Array, Bool, Float, Int_, Integer, Parameters

__all__ = [
    "NoStopTermination",
    "PositionalTermination",
    "RetrievalDependentTermination",
    "SupportRatioTermination",
    "SupportsNoStop",
    "SupportsPositional",
    "SupportsRetrievalDependent",
    "SupportsSupportRatio",
]


class SupportsNoStop(Protocol):
    """Availability information needed to stop when retrieval is exhausted.

    Models and lightweight views satisfy these read-only requirements
    structurally; they do not need to inherit from this protocol.
    """

    @property
    def recallable(self) -> Bool[Array, " candidates"]:
        """Indicate which candidates are currently available."""
        ...

    @property
    def is_active(self) -> Bool[Array, ""]:
        """Indicate whether retrieval is still active."""
        ...


class SupportsPositional(SupportsNoStop, Protocol):
    """Availability and recall count needed for positional stopping."""

    @property
    def recall_total(self) -> Integer[Array, ""]:
        """Return the number of recall events so far."""
        ...


class SupportsRetrievalDependent(SupportsNoStop, Protocol):
    """Input requirements of the legacy STOP-item policy.

    ``item_count`` must identify an explicit STOP candidate, and evaluating its
    outcome probability must not call this termination policy again. Structural
    typing cannot establish either condition. Ordinary CMR models use
    ``item_count`` for their final item and do not meet these semantic conditions.
    """

    @property
    def item_count(self) -> int:
        """Return the index this policy interprets as its STOP candidate."""
        ...

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Evaluate the requested outcome independently of this stopping rule."""
        ...


class SupportsSupportRatio(SupportsNoStop, Protocol):
    """Candidate information needed by the existing support-ratio rule.

    ``studied``, ``recallable``, and candidate activations must refer to the same
    candidate space. The rule treats studied, unavailable candidates as already
    recalled; models using another availability convention need a different
    interpretation before this rule can be used with them.
    """

    @property
    def studied(self) -> Bool[Array, " candidates"]:
        """Indicate which candidates were studied."""
        ...

    def candidate_activations(
        self, candidates: Bool[Array, " candidates"]
    ) -> Float[Array, " candidates"]:
        """Return support for the selected candidates, including unavailable ones.

        Apply the model's retrieval-support calculation to this mask without
        additionally suppressing candidates using ``recallable``.
        """
        ...


class NoStopTermination(Pytree):
    """Termination probability is always zero as long as recalls remain."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        pass

    def stop_probability(self, model: SupportsNoStop) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            lambda: jnp.array(0.0),
        )


class PositionalTermination(Pytree):
    """Termination probability is an exponential function of recall position."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        scale = parameters["stop_probability_scale"]
        growth = parameters["stop_probability_growth"]
        self._stop_probability = scale * jnp.exp(jnp.arange(list_length) * growth)

    def stop_probability(self, model: SupportsPositional) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        probability_dtype = self._stop_probability.dtype
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.ones((), dtype=probability_dtype),
            lambda: jnp.minimum(
                1.0 - float(FLOAT32_EPS) * total_recallable.astype(probability_dtype),
                self._stop_probability[model.recall_total],
            ),
        )


class RetrievalDependentTermination(Pytree):
    """Legacy STOP-item rule; requires ``SupportsRetrievalDependent`` semantics.

    Ordinary CMR outcome probabilities depend on termination, so attaching this
    rule to ordinary CMR creates recursion even though the method names match.
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        pass

    def stop_probability(self, model: SupportsRetrievalDependent) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)

        def compute_stop_probability() -> Float[Array, ""]:
            probability = model.outcome_probability(model.item_count)
            return jnp.minimum(
                1.0 - float(FLOAT32_EPS) * total_recallable.astype(probability.dtype),
                probability,
            )

        # Inspect the callback's output dtype without executing it when stopped.
        probability_dtype = eval_shape(compute_stop_probability).dtype
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.ones((), dtype=probability_dtype),
            compute_stop_probability,
        )


class SupportRatioTermination(Pytree):
    """Termination probability is based on ratio of recalled to not recalled support."""

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]

    def stop_probability(self, model: SupportsSupportRatio) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)

        def compute_stop_probability() -> Float[Array, ""]:
            """Returns stop probability derived from support ratios."""
            recalled_mask = jnp.logical_and(model.studied, ~model.recallable)
            not_recalled_mask = jnp.logical_and(model.studied, model.recallable)
            recalled_support = jnp.sum(model.candidate_activations(recalled_mask))
            not_recalled_support = jnp.sum(
                model.candidate_activations(not_recalled_mask)
            )
            support_ratio = lax.cond(
                recalled_support > 0.0,
                lambda: not_recalled_support / recalled_support,
                lambda: jnp.full_like(recalled_support, jnp.inf),
            )
            stop_probability = self.scale + jnp.exp(-self.growth * support_ratio)
            return jnp.minimum(
                stop_probability,
                1.0 - float(FLOAT32_EPS) * total_recallable.astype(stop_probability.dtype),
            )

        # Infer precision through the model's support calculation, without
        # evaluating support ratios for an inactive or exhausted model.
        probability_dtype = eval_shape(compute_stop_probability).dtype
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.ones((), dtype=probability_dtype),
            compute_stop_probability,
        )
