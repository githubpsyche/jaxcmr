"""Use retrieval evidence to decide whether recall continues. Each studied item
contributes its current support to one of two sums, according to whether it
can still be recalled. More available support relative to unavailable support
makes stopping less likely.

The two response policies use the same stopping probability. If recall
continues, RawSupportRatioSoftmaxChoice applies a softmax to the available
items' evidence. RawSupportRatioRelativeChoice uses the model's own candidate
weights, including its choice sensitivity and any semantic weighting.
"""

from typing import Protocol

import jax.numpy as jnp
from jax.nn import softmax

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.response import (
    SupportsCandidateWeights,
    SupportsStudiedEvidence,
    _probabilities_with_stop,
)
from jaxcmr.components.termination import support_ratio_stop_probability
from jaxcmr.typing import Array, Bool, Bool_, Float, Float_, Parameters


def raw_support_ratio_stop_probability(
    evidence: Float[Array, " items"],
    studied: Bool[Array, " items"],
    recallable: Bool[Array, " items"],
    *,
    scale: Float_,
    growth: Float_,
    is_active: Bool_,
) -> Float[Array, ""]:
    """Calculate STOP from summed item evidence before the item-choice transform.

    Parameters
    ----------
    evidence : floating array
        Current retrieval support per item. Each studied item contributes at
        least 1e-6. Padding and other unstudied items contribute zero.
    studied, recallable : Boolean arrays
        Masks in evidence order. An item still eligible for repeated recall
        contributes to the available sum even after it has been recalled.
    scale : float
        Baseline stopping probability before any studied item is unavailable.
    growth : float
        Positive coefficient of the available-to-unavailable support ratio.
    is_active : bool
        Whether retrieval is in progress.

    Returns
    -------
    floating scalar
        For ratio R, the stopping probability is scale + exp(-growth * R),
        capped to leave probability for eligible items. Retrieval stops with
        certainty on an inactive or exhausted trial.
    """
    support = jnp.maximum(evidence, 1e-6)
    available = studied & recallable
    unavailable = studied & ~recallable
    return support_ratio_stop_probability(
        jnp.sum(jnp.where(available, support, 0)),
        jnp.sum(jnp.where(unavailable, support, 0)),
        scale=scale,
        growth=growth,
        is_active=is_active,
        available_count=available.sum(),
    )


class RawSupportRatioSoftmaxChoice(Pytree):
    """Stop by raw support, then choose an available item by softmax.

    The policy reads ``stop_probability_scale`` and ``stop_probability_growth``.
    Conditional on continuing, item i receives weight exp(evidence[i]) using
    the model's original evidence, including values below the stopping floor.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]

    def outcome_probabilities(self, model: SupportsStudiedEvidence) -> Float[Array, " outcomes"]:
        evidence = model.item_evidence()
        stop = raw_support_ratio_stop_probability(
            evidence, model.studied, model.recallable,
            scale=self.scale, growth=self.growth, is_active=model.is_active,
        )
        choice = softmax(evidence, where=model.studied & model.recallable)
        return jnp.append(stop, (1 - stop) * choice)


class SupportsStudiedEvidenceAndWeights(SupportsStudiedEvidence, SupportsCandidateWeights, Protocol):
    """Supply raw evidence for stopping and candidate weights for item choice."""


class RawSupportRatioRelativeChoice(Pytree):
    """Stop by raw support, then use the model's relative-power item choice.

    The stopping parameters are the same as for RawSupportRatioSoftmaxChoice.
    Only item choice uses ``choice_sensitivity``. The model supplies the
    weights, preserving its treatment of semantic support.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]
        self.sensitivity = parameters["choice_sensitivity"]

    def outcome_probabilities(
        self, model: SupportsStudiedEvidenceAndWeights
    ) -> Float[Array, " outcomes"]:
        stop = raw_support_ratio_stop_probability(
            model.item_evidence(), model.studied, model.recallable,
            scale=self.scale, growth=self.growth, is_active=model.is_active,
        )
        weights = model.candidate_weights(model.studied & model.recallable, self.sensitivity)
        return _probabilities_with_stop(stop, weights)
