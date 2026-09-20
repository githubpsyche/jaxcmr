"""Termination policies accept only the capabilities they actually use."""

from collections.abc import Sequence

import jax.numpy as jnp

from jaxcmr.components.termination import (
    NoStopTermination,
    PositionalTermination,
    RetrievalDependentTermination,
    SupportRatioTermination,
)
from jaxcmr.typing import Array, Bool, Float, Int_


class _RecallState:
    """Availability-only input, with no memory-search methods."""

    def __init__(self, recallable: Sequence[bool], is_active: bool = True):
        self.recallable = jnp.array(recallable, dtype=bool)
        self.is_active = jnp.array(is_active)


class _PositionalState(_RecallState):
    """Availability and recall count, with no support calculation."""

    def __init__(
        self,
        recallable: Sequence[bool],
        is_active: bool = True,
        recall_total: int = 0,
    ):
        super().__init__(recallable, is_active)
        self.recall_total = jnp.array(recall_total, dtype=int)


class _StopState(_RecallState):
    """Explicit STOP probability independent of the termination policy."""

    def __init__(self, recallable: Sequence[bool]):
        super().__init__(recallable)
        self.item_count = len(recallable)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        return jnp.where(choice == self.item_count, 0.1, 0.0)


class _SupportState(_RecallState):
    """Candidate supports, with no context or memory matrices."""

    def __init__(
        self,
        recallable: Sequence[bool],
        studied: Sequence[bool] | None = None,
        activations: Sequence[float] | None = None,
    ):
        super().__init__(recallable)
        self.studied = (
            jnp.ones_like(self.recallable)
            if studied is None
            else jnp.array(studied, dtype=bool)
        )
        self._activations = (
            jnp.zeros(len(recallable))
            if activations is None
            else jnp.array(activations, dtype=float)
        )

    def candidate_activations(
        self, candidates: Bool[Array, " candidates"]
    ) -> Float[Array, " candidates"]:
        return self._activations * candidates


# ---------------------------------------------------------------------------
# NoStopTermination
# ---------------------------------------------------------------------------


def test_no_stop_returns_zero_when_items_available():
    """Behavior: ``NoStopTermination`` yields zero while items remain.

    Given:
      - An active model with recallable items.
    When:
      - ``stop_probability`` is called.
    Then:
      - Returns 0.0.
    Why this matters:
      - Models without explicit stopping never voluntarily terminate.
    """
    # Arrange / Given
    policy = NoStopTermination(3, {})
    model = _RecallState(recallable=[True, True, False])

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert jnp.isclose(p_stop, 0.0).item()


def test_no_stop_returns_one_when_no_items_available():
    """Behavior: ``NoStopTermination`` forces stop when pool exhausted.

    Given:
      - An active model with no recallable items.
    When:
      - ``stop_probability`` is called.
    Then:
      - Returns 1.0.
    Why this matters:
      - Recall must terminate when no items can be retrieved.
    """
    # Arrange / Given
    policy = NoStopTermination(3, {})
    model = _RecallState(recallable=[False, False, False])

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert jnp.isclose(p_stop, 1.0).item()


# ---------------------------------------------------------------------------
# PositionalTermination
# ---------------------------------------------------------------------------


def test_positional_returns_small_probability_at_start():
    """Behavior: ``PositionalTermination`` yields a small stop probability early.

    Given:
      - An active model at recall position 0.
    When:
      - ``stop_probability`` is called.
    Then:
      - The probability is small (close to scale parameter).
    Why this matters:
      - Early in recall, stopping should be unlikely.
    """
    # Arrange / Given
    params = {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    policy = PositionalTermination(5, params)
    model = _PositionalState(recallable=[True] * 5, recall_total=0)

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert float(p_stop) < 0.1
    assert float(p_stop) > 0.0


def test_positional_increases_with_recall_position():
    """Behavior: ``PositionalTermination`` probability grows with position.

    Given:
      - An active model at increasing recall positions.
    When:
      - ``stop_probability`` is compared across positions.
    Then:
      - Later positions yield higher stop probabilities.
    Why this matters:
      - Models should stop more readily as recall progresses.
    """
    # Arrange / Given
    params = {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    policy = PositionalTermination(5, params)

    # Act / When
    p_early = policy.stop_probability(
        _PositionalState(recallable=[True] * 5, recall_total=0)
    )
    p_late = policy.stop_probability(
        _PositionalState(recallable=[True] * 5, recall_total=4)
    )

    # Assert / Then
    assert float(p_late) > float(p_early)


def test_positional_returns_one_when_inactive():
    """Behavior: ``PositionalTermination`` forces stop when model inactive.

    Given:
      - An inactive model.
    When:
      - ``stop_probability`` is called.
    Then:
      - Returns 1.0.
    Why this matters:
      - An inactive model has already terminated.
    """
    # Arrange / Given
    params = {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    policy = PositionalTermination(3, params)
    model = _PositionalState(recallable=[True, True, True], is_active=False)

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert jnp.isclose(p_stop, 1.0).item()


# ---------------------------------------------------------------------------
# RetrievalDependentTermination
# ---------------------------------------------------------------------------


def test_retrieval_dependent_uses_model_outcome_probability():
    """Behavior: ``RetrievalDependentTermination`` delegates to model.

    Given:
      - An active model whose ``outcome_probability(item_count)`` returns 0.1.
    When:
      - ``stop_probability`` is called.
    Then:
      - The result equals the model's stop-item probability.
    Why this matters:
      - Confirms termination is driven by the retrieval competition.
    """
    # Arrange / Given
    policy = RetrievalDependentTermination(3, {})
    model = _StopState(recallable=[True, True, True])

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert jnp.isclose(p_stop, 0.1).item()


# ---------------------------------------------------------------------------
# SupportRatioTermination
# ---------------------------------------------------------------------------


def test_support_ratio_returns_one_when_no_items_recallable():
    """Behavior: ``SupportRatioTermination`` forces stop when pool exhausted.

    Given:
      - A model with no recallable items.
    When:
      - ``stop_probability`` is called.
    Then:
      - Returns 1.0.
    Why this matters:
      - Mandatory termination when nothing can be recalled.
    """
    # Arrange / Given
    params = {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    policy = SupportRatioTermination(3, params)
    model = _SupportState(recallable=[False, False, False])

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    assert jnp.isclose(p_stop, 1.0).item()


def test_support_ratio_uses_candidate_activations_without_mcf():
    """Behavior: ``SupportRatioTermination`` uses model candidate activations.

    Given:
      - A model with one recalled and two recallable studied items.
      - Candidate activations but no ``mcf`` attribute.
    When:
      - ``stop_probability`` is called.
    Then:
      - The stop probability is computed from candidate activations.
    Why this matters:
      - Support-ratio stopping must reflect what the model treats as
        retrieval support, not a specific internal memory matrix.
    """
    # Arrange / Given
    params = {"stop_probability_scale": 0.01, "stop_probability_growth": 0.5}
    policy = SupportRatioTermination(3, params)
    model = _SupportState(
        recallable=[False, True, True],
        activations=[2.0, 4.0, 6.0],
    )

    # Act / When
    p_stop = policy.stop_probability(model)

    # Assert / Then
    expected = 0.01 + jnp.exp(-0.5 * 5.0)
    assert jnp.isclose(p_stop, expected).item()
