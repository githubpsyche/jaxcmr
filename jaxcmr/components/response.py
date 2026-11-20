"""A response distribution gives the probability of STOP followed by each item.

In ordinary CMR the model weights each item by its context match. A semantic
model can add the last recalled item's semantic associations. A positional
model can sum the weights of several study positions for the same item.
PositionalStop and RecallUntilExhausted use the weights supplied by the model.

Other rules compete STOP directly against raw evidence. They require a model
that supplies evidence in those units. LearnedStopCompetition also requires
the model to encode a terminal item during study.
"""

from typing import Protocol

import jax.numpy as jnp
from jax import lax, vmap
from jax.nn import softmax

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.components.termination import (
    SupportsNoStop,
    SupportsPositional,
    positional_stop_probability,
    stop_when_exhausted,
    support_ratio_stop_probability,
)
from jaxcmr.math import FLOAT32_EPS, relative_power_scale
from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Integer,
    Parameters,
)


class SupportsItemEvidence(SupportsNoStop, Protocol):
    """Supply item evidence in the model's units before the item-choice transform."""

    def item_evidence(self) -> Float[Array, " items"]: ...


class SupportsCandidateWeights(SupportsNoStop, Protocol):
    """Return item weights for a selection of retrieval candidates.

    The mask indexes the same candidates as ``recallable``. Results are in
    recorded item-code order. Unstudied candidates contribute zero weight.
    """

    def candidate_weights(
        self, candidates: Bool[Array, " candidates"], sensitivity: Float_
    ) -> Float[Array, " items"]:
        """Weight the selected candidates and combine contributions to each item.

        Parameters
        ----------
        candidates : Boolean array
            A mask in the model's retrieval order. Positions in a positional
            model can refer to several occurrences of the same item.
        sensitivity : float
            The choice exponent supplied by the response policy.

        Returns
        -------
        floating array
            One weight per item code, with zero for an item that has no eligible
            occurrence. Within a single route of a blended model, each entry
            instead identifies a candidate on that route.
        """
        ...


class SupportsWeightsAndRecallCount(
    SupportsCandidateWeights, SupportsPositional, Protocol
):
    """Supply candidate weights and the number of recalls already made."""


def pool_position_weights(
    weights: Float[Array, " positions"], study_items: Integer[Array, " positions"]
) -> Float[Array, " items"]:
    """Sum position weights by recorded item code.

    Parameters
    ----------
    weights : floating array
        One weight or probability per study position.
    study_items : integer array
        One-based item codes at those positions. Zero marks padding.

    Returns
    -------
    floating array
        Entry ``i - 1`` sums the contributions of every occurrence of item ``i``.
        The output has one slot per possible item, including absent items.
    """
    return lax.map(
        lambda code: jnp.sum(jnp.where(study_items == code, weights, 0)),
        jnp.arange(1, study_items.size + 1),
    )


def relative_choice_weights(
    evidence: Float[Array, " items"],
    candidates: Bool[Array, " items"],
    sensitivity: Float_,
) -> Float[Array, " items"]:
    """Weight each candidate by its evidence relative to the strongest candidate.

    Parameters
    ----------
    evidence : floating array
        Memory support in candidate order.
    candidates : Boolean array
        Candidates included in this choice.
    sensitivity : float
        Exponent applied to the relative evidence.

    Returns
    -------
    floating array
        Each selected candidate receives its scaled evidence plus FLOAT32_EPS.
        When all evidence is zero, selected candidates receive equal weights.
        Excluded candidates receive zero."""
    return (
        relative_power_scale(evidence * candidates, sensitivity) + FLOAT32_EPS
    ) * candidates


def _probabilities_with_stop(
    stop: Float[Array, ""], weights: Float[Array, " items"]
) -> Float[Array, " outcomes"]:
    total = jnp.sum(weights)
    return jnp.append(stop, (1 - stop) * weights / jnp.where(total == 0, 1, total))


class SupportsEvidenceAndRecallCount(
    SupportsItemEvidence, SupportsPositional, Protocol
):
    """Supply item evidence, availability and the number of recalls already made."""


class PositionalStop(Pytree):
    """Choose STOP by output position and divide the remaining probability among items.

    After k recalls, the scheduled stopping probability is
    ``stop_probability_scale * exp(k * stop_probability_growth)``, subject
    to the positional probability ceiling. Available items share the remaining
    probability in proportion to the model’s candidate weights at ``choice_sensitivity``.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.sensitivity = parameters["choice_sensitivity"]
        self.schedule = parameters["stop_probability_scale"] * jnp.exp(
            jnp.arange(list_length) * parameters["stop_probability_growth"]
        )

    def outcome_probabilities(
        self, model: SupportsWeightsAndRecallCount
    ) -> Float[Array, " outcomes"]:
        stop = positional_stop_probability(
            self.schedule,
            model.recall_total,
            is_active=model.is_active,
            available_count=jnp.sum(model.recallable),
        )
        weights = model.candidate_weights(model.recallable, self.sensitivity)
        return _probabilities_with_stop(stop, weights)


class RecallUntilExhausted(Pytree):
    """Choose among available items in proportion to their model-supplied weights.

    The policy passes ``choice_sensitivity`` to the model. STOP has probability
    zero while an active trial has an available item, and one otherwise.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.sensitivity = parameters["choice_sensitivity"]

    def outcome_probabilities(
        self, model: SupportsCandidateWeights
    ) -> Float[Array, " outcomes"]:
        stop = stop_when_exhausted(model.is_active, jnp.sum(model.recallable))
        weights = model.candidate_weights(model.recallable, self.sensitivity)
        return _probabilities_with_stop(stop, weights)


def _scheduled_competition(
    model: SupportsEvidenceAndRecallCount,
    stop_log_weight: Float[Array, ""],
) -> Float[Array, " outcomes"]:
    item_logits = jnp.where(
        model.is_active & model.recallable, model.item_evidence(), -jnp.inf
    )
    return softmax(jnp.append(stop_log_weight, item_logits))


class ExponentialStopCompetition(Pytree):
    """Compete exp(stop_log_weight + stop_weight_growth * k) against exp(item evidence).

    k counts previous recalls. The intercept is the log weight of STOP before
    the first recall. No maximum normalization or CMR choice exponent is applied.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.intercept = parameters["stop_log_weight"]
        self.growth = parameters["stop_weight_growth"]

    def outcome_probabilities(
        self, model: SupportsEvidenceAndRecallCount
    ) -> Float[Array, " outcomes"]:
        return _scheduled_competition(
            model, self.intercept + self.growth * model.recall_total
        )


class PowerLawStopCompetition(Pytree):
    """Compete exp(stop_log_weight) * (1 + k)**stop_weight_growth against exp(item evidence).

    The first response uses k = 0. Item evidence uses the same units as in
    exponential competition.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.intercept = parameters["stop_log_weight"]
        self.growth = parameters["stop_weight_growth"]

    def outcome_probabilities(
        self, model: SupportsEvidenceAndRecallCount
    ) -> Float[Array, " outcomes"]:
        # Match the fitted parameters' precision when taking the log of an integer count.
        count = jnp.asarray(model.recall_total, dtype=jnp.result_type(self.growth, 1.0))
        return _scheduled_competition(
            model, self.intercept + self.growth * jnp.log1p(count)
        )


class SupportsStudiedEvidence(SupportsItemEvidence, Protocol):
    """Supply evidence for both available and unavailable studied items."""

    @property
    def studied(self) -> Bool[Array, " items"]: ...


class RelativePowerSupportRatio(Pytree):
    """Reproduce stopping based on separately transformed candidate weights.

    The model weights the two groups separately at ``choice_sensitivity``.
    In ordinary CMR, each group scales its evidence by its own maximum.
    The ratio of the two totals determines stopping. The same available weights determine item
    choice when recall continues.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.sensitivity = parameters["choice_sensitivity"]
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]

    def outcome_probabilities(
        self, model: SupportsCandidateWeights
    ) -> Float[Array, " outcomes"]:
        # Map over the two masks so shared memory probes are evaluated once.
        available_weights, unavailable_weights = vmap(
            model.candidate_weights, in_axes=(0, None)
        )(jnp.stack((model.recallable, ~model.recallable)), self.sensitivity)
        stop = support_ratio_stop_probability(
            jnp.sum(available_weights),
            jnp.sum(unavailable_weights),
            scale=self.scale,
            growth=self.growth,
            is_active=model.is_active,
            available_count=jnp.sum(model.recallable),
        )
        return _probabilities_with_stop(stop, available_weights)


class ExponentialEvidenceSupportRatio(Pytree):
    """Use exp(item evidence) for both the support ratio and conditional item choice.

    The numerator sums weights of available studied items. The denominator sums
    weights of unavailable studied items. Before any item becomes unavailable,
    STOP has probability ``stop_probability_scale`` (subject to the usual cap).
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]

    def outcome_probabilities(
        self, model: SupportsStudiedEvidence
    ) -> Float[Array, " outcomes"]:
        evidence = model.item_evidence()
        evidence = evidence.astype(jnp.result_type(evidence, self.scale, self.growth))
        available = model.studied & model.recallable
        unavailable = model.studied & ~model.recallable

        def while_active():
            # Subtract one common maximum so the ratio retains its absolute
            # scale across the two groups while exponentiation stays finite.
            maximum = jnp.max(jnp.where(model.studied, evidence, -jnp.inf))
            weights = jnp.exp(jnp.where(model.studied, evidence - maximum, -jnp.inf))
            stop = support_ratio_stop_probability(
                jnp.sum(jnp.where(available, weights, 0)),
                jnp.sum(jnp.where(unavailable, weights, 0)),
                scale=self.scale,
                growth=self.growth,
                is_active=model.is_active,
                available_count=jnp.sum(available),
            )
            choice = softmax(jnp.where(available, evidence, -jnp.inf))
            return jnp.append(stop, (1 - stop) * choice)

        return lax.cond(
            model.is_active & jnp.any(available),
            while_active,
            lambda: jnp.append(
                jnp.ones((), dtype=evidence.dtype), jnp.zeros_like(evidence)
            ),
        )


class SupportsTerminalItem(SupportsNoStop, Protocol):
    """Supply raw evidence for the learned terminal candidate followed by the items."""

    def terminal_and_item_evidence(self) -> Float[Array, " outcomes"]: ...


class LearnedStopCompetition(Pytree):
    """Choose among a learned terminal item and eligible study items in proportion to weight.

    The model encodes the terminal item at list end. While retrieval remains
    possible, the CRU hybrid reserves at least FLOAT32_EPS mass per available
    item in total. Exhausted or inactive trials stop with probability one.
    """

    def __init__(self, list_length: int, parameters: Parameters) -> None:
        self.sensitivity = parameters["choice_sensitivity"]

    def outcome_probabilities(
        self, model: SupportsTerminalItem
    ) -> Float[Array, " outcomes"]:
        weights = relative_choice_weights(
            model.terminal_and_item_evidence(),
            jnp.append(True, model.recallable),
            self.sensitivity,
        )
        item_weights = weights[1:]
        item_total = jnp.sum(item_weights)
        total = weights[0] + item_total
        count = jnp.sum(model.recallable).astype(weights.dtype)
        stop = jnp.minimum(
            weights[0] / jnp.where(total == 0, 1, total), 1 - float(FLOAT32_EPS) * count
        )
        stop = jnp.where(model.is_active & (count > 0), stop, 1)
        return _probabilities_with_stop(stop, item_weights)
