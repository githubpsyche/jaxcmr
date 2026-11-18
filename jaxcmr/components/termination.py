"""The probability of stopping depends on which candidates remain available.
A model can stop only on exhaustion, use a schedule indexed by recall count,
or compare support for available and unavailable studied candidates. Each
calculation returns one if retrieval has ended or no candidates remain.

A model with a fixed stopping rule can call the corresponding function
directly. To select a rule when configuring CMR, pass its class to the model
factory. Each class specifies what it reads from the model. The legacy
STOP-item rule requires a model that scores STOP as a separate candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from jax import eval_shape, lax
from jax import numpy as jnp

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.math import FLOAT32_EPS
from jaxcmr.typing import Array, Bool, Bool_, Float, Float_, Int_, Integer, Parameters

__all__ = [
    "NoStopTermination",
    "PositionalTermination",
    "RetrievalDependentTermination",
    "SupportRatioTermination",
    "SupportsNoStop",
    "SupportsPositional",
    "SupportsRetrievalDependent",
    "SupportsSupportRatio",
    "positional_stop_probability",
    "stop_when_exhausted",
    "support_ratio_stop_probability",
]


def stop_when_exhausted(is_active: Bool_, available_count: Int_) -> Float[Array, ""]:
    """Stop when retrieval has ended or there are no available candidates.

    Parameters
    ----------
    is_active : bool
        Whether the trial is still in retrieval.
    available_count : int
        Number of candidates that the model can still recall.

    Returns
    -------
    Float[Array, ""]
        One if retrieval has ended or the count is zero, otherwise zero.
    """
    return jnp.where(jnp.logical_and(is_active, available_count != 0), 0.0, 1.0)


class SupportsNoStop(Protocol):
    """Availability information needed to stop when retrieval is exhausted.

    ``is_active`` indicates whether retrieval is in progress. Each true entry
    in ``recallable`` marks a candidate the model can still retrieve.
    """

    @property
    def recallable(self) -> Bool[Array, " candidates"]:
        """Indicate which candidates are currently available."""
        ...

    @property
    def is_active(self) -> Bool[Array, ""]:
        """Indicate whether retrieval is still active."""
        ...


class NoStopTermination(Pytree):
    """Continue until the model becomes inactive or exhausts its available candidates.

    Requires ``is_active`` and a Boolean ``recallable`` mask. The constructor
    accepts the standard list length and parameter mapping and reads no values
    from them. [Model requirements](#jaxcmr.components.termination.SupportsNoStop)
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        pass

    def stop_probability(self, model: SupportsNoStop) -> Float[Array, ""]:
        return stop_when_exhausted(model.is_active, jnp.sum(model.recallable))


def _capped_stop_probability(
    is_active: Bool_,
    available_count: Int_,
    probability_while_active: Callable[[], Float[Array, ""]],
) -> Float[Array, ""]:
    """Evaluate and cap a rule only while retrieval can continue."""

    def capped_probability() -> Float[Array, ""]:
        probability = probability_while_active()
        count = jnp.asarray(available_count, dtype=probability.dtype)
        return jnp.minimum(probability, 1.0 - float(FLOAT32_EPS) * count)

    # Tracing the return dtype lets an exhausted trial skip the support calculation.
    dtype = eval_shape(probability_while_active).dtype
    return lax.cond(
        jnp.logical_and(is_active, available_count != 0),
        capped_probability,
        lambda: jnp.ones((), dtype=dtype),
    )


def positional_stop_probability(
    schedule: Float[Array, " recall_counts"],
    recall_total: Int_,
    *,
    is_active: Bool_,
    available_count: Int_,
) -> Float[Array, ""]:
    """Look up the probability of stopping after the given number of recalls.

    Parameters
    ----------
    schedule : floating array
        Nonempty sequence of stopping probabilities. Entry zero applies before
        the first recall. Counts beyond the schedule use the last entry.
    recall_total : int
        Number of items already recalled.
    is_active : bool
        Whether the trial is still in retrieval.
    available_count : int
        Number of candidates that the model can still recall.

    Returns
    -------
    Float[Array, ""]
        One if retrieval has ended or no candidates remain. Otherwise the
        scheduled value capped at ``1 - FLOAT32_EPS * available_count``,
        in the schedule's floating-point precision.
    """
    return _capped_stop_probability(
        is_active, available_count, lambda: schedule[recall_total]
    )


class SupportsPositional(SupportsNoStop, Protocol):
    """Availability and recall count needed for positional stopping.

    In addition to ``is_active`` and ``recallable``, the model supplies
    ``recall_total``, the number of recall events so far.
    """

    @property
    def recall_total(self) -> Integer[Array, ""]:
        """Return the number of recall events so far."""
        ...


class PositionalTermination(Pytree):
    """Set stopping probability from the number of items recalled so far.

    At recall count ``r``, the probability is
    ``stop_probability_scale * exp(r * stop_probability_growth)``, capped at
    ``1 - FLOAT32_EPS * number_available``. The first decision uses ``r = 0``.
    The schedule ends at ``r = list_length - 1``. Higher counts use its last
    value. The model supplies ``is_active``, the candidate mask ``recallable``,
    and the recall count ``recall_total``. Ordinary CMR uses this rule by default.
    [Model requirements](#jaxcmr.components.termination.SupportsPositional)

    Parameters
    ----------
    list_length : int
        Number of positions in the stopping schedule, starting at count zero.
    parameters : Parameters
        ``stop_probability_scale`` gives the initial value before the cap.
        ``stop_probability_growth`` controls its change per recall.
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        scale = parameters["stop_probability_scale"]
        growth = parameters["stop_probability_growth"]
        self._stop_probability = scale * jnp.exp(jnp.arange(list_length) * growth)

    def stop_probability(self, model: SupportsPositional) -> Float[Array, ""]:
        return positional_stop_probability(
            self._stop_probability, model.recall_total,
            is_active=model.is_active, available_count=jnp.sum(model.recallable),
        )


def _support_ratio_probability(
    available_support: Float_,
    unavailable_support: Float_,
    scale: Float_,
    growth: Float_,
) -> Float[Array, ""]:
    """Apply the support-ratio stopping equation before the probability ceiling."""
    dtype = jnp.result_type(available_support, unavailable_support, 0.0)
    ratio = lax.cond(
        unavailable_support > 0,
        lambda: jnp.divide(available_support, unavailable_support),
        lambda: jnp.asarray(jnp.inf, dtype=dtype),
    )
    return scale + jnp.exp(-growth * ratio)


def support_ratio_stop_probability(
    available_support: Float_,
    unavailable_support: Float_,
    *,
    scale: Float_,
    growth: Float_,
    is_active: Bool_,
    available_count: Int_,
) -> Float[Array, ""]:
    """Calculate stopping probability from two totals of retrieval support.

    Parameters
    ----------
    available_support : float
        Total support for studied candidates that are available for recall.
    unavailable_support : float
        Total support for studied candidates that are unavailable for recall.
    scale : float
        Additive term in ``scale + exp(-growth * ratio)``. The ratio is available
        support divided by unavailable support, or infinity when the latter is zero.
    growth : float
        Coefficient of the support ratio in the exponential.
    is_active : bool
        Whether the trial is still in retrieval.
    available_count : int
        Number of candidates that the model can still recall.

    Returns
    -------
    Float[Array, ""]
        One if retrieval has ended or no candidates remain. Otherwise the
        result of the support-ratio equation, capped at
        ``1 - FLOAT32_EPS * available_count``. The ceiling uses float32
        machine epsilon even for float64 probabilities.
    """
    return _capped_stop_probability(
        is_active,
        available_count,
        lambda: _support_ratio_probability(
            available_support, unavailable_support, scale, growth
        ),
    )


class SupportsSupportRatio(SupportsNoStop, Protocol):
    """Candidate information needed by the existing support-ratio rule.

    ``studied``, ``recallable``, and candidate activations must refer to the same
    candidate space. The rule scores the available and unavailable subsets of
    studied candidates separately. A recalled item can still belong to the
    available subset if the model permits repeated recall.
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


class SupportRatioTermination(Pytree):
    """Compare support for available and unavailable studied candidates.

    The model supplies ``is_active``, the masks ``studied`` and ``recallable``,
    and ``candidate_activations(mask)``. Both masks and the returned scores
    refer to the same candidates. The model must be able to score unavailable
    candidates too. The rule scores each group separately and sums its support.

    In ordinary CMR, recalling an item makes it unavailable when repeated
    recalls are disabled. When repeats are enabled, a recalled item stays in
    the available group and contributes to its support.
    [Model requirements](#jaxcmr.components.termination.SupportsSupportRatio)

    Parameters
    ----------
    list_length : int
        Accepted as part of the stopping-rule constructor interface.
    parameters : Parameters
        ``stop_probability_scale`` sets the additive term.
        ``stop_probability_growth`` sets the coefficient of the support ratio.
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        self.scale = parameters["stop_probability_scale"]
        self.growth = parameters["stop_probability_growth"]

    def stop_probability(self, model: SupportsSupportRatio) -> Float[Array, ""]:
        def probability_while_active() -> Float[Array, ""]:
            unavailable = model.studied & ~model.recallable
            available = model.studied & model.recallable
            unavailable_support = jnp.sum(model.candidate_activations(unavailable))
            available_support = jnp.sum(model.candidate_activations(available))
            return _support_ratio_probability(
                available_support, unavailable_support, self.scale, self.growth
            )

        return _capped_stop_probability(
            model.is_active, jnp.sum(model.recallable), probability_while_active
        )


class SupportsRetrievalDependent(SupportsNoStop, Protocol):
    """Input requirements of the legacy STOP-item policy.

    ``item_count`` must identify an explicit STOP candidate, and evaluating its
    outcome probability must not call this termination policy again. Structural
    typing cannot establish either condition. Ordinary CMR models use
    ``item_count`` as the highest local item code.
    """

    @property
    def item_count(self) -> int:
        """Return the index this policy interprets as its STOP candidate."""
        ...

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Evaluate the requested outcome independently of this stopping rule."""
        ...


class RetrievalDependentTermination(Pytree):
    """Use a dedicated STOP candidate's independently calculated probability.

    The model supplies ``is_active`` and a Boolean ``recallable`` mask. It must
    score ``outcome_probability(item_count)`` as STOP without
    calling this policy. In ordinary CMR, ``item_count`` is the highest local
    item code and scoring an outcome calls the stopping rule again.

    The rule reads no stopping parameters. While recall can continue, the
    returned probability is capped at ``1 - FLOAT32_EPS * number_available``.
    [Model requirements](#jaxcmr.components.termination.SupportsRetrievalDependent)
    """

    def __init__(
        self,
        list_length: int,
        parameters: Parameters,
    ) -> None:
        pass

    def stop_probability(self, model: SupportsRetrievalDependent) -> Float[Array, ""]:
        return _capped_stop_probability(
            model.is_active, jnp.sum(model.recallable),
            lambda: model.outcome_probability(model.item_count),
        )
