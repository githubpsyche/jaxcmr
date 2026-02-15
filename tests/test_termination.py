import jax.numpy as jnp

from jaxcmr.components.termination import (
    NoStopTermination,
    PositionalTermination,
    RetrievalDependentTermination,
    SupportRatioTermination,
)


class _FakeContext:
    """Minimal context satisfying the Context protocol."""

    def __init__(self, size: int = 3):
        self.size = size
        self.state = jnp.zeros(size)
        self.initial_state = jnp.zeros(size)

    def integrate(self, context_input, drift_rate):  # noqa: D401
        return self


class _FakeModel:
    """Minimal model satisfying the MemorySearch protocol."""

    def __init__(
        self,
        recallable,
        is_active=True,
        recall_total=0,
        item_count=3,
        studied=None,
    ):
        self.recallable = jnp.array(recallable, dtype=bool)
        self.is_active = jnp.array(is_active)
        self.recall_total = jnp.array(recall_total, dtype=int)
        self.item_count = item_count
        self.studied = studied if studied is not None else jnp.ones_like(self.recallable)
        self.study_index = jnp.array(item_count, dtype=int)
        self.context = _FakeContext(item_count)

    def experience(self, choice):  # noqa: D401
        return self

    def start_retrieving(self):  # noqa: D401
        return self

    def retrieve(self, choice):  # noqa: D401
        return self

    def activations(self):  # noqa: D401
        return jnp.zeros(self.item_count)

    def outcome_probability(self, choice):
        return jnp.array(0.1)

    def outcome_probabilities(self):  # noqa: D401
        return jnp.ones(self.item_count + 1) / (self.item_count + 1)


class _FakeModelWithMcf(_FakeModel):
    """Fake model with an mcf.probe interface for SupportRatioTermination."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcf = self
        self.context = self

    @property
    def state(self):
        return jnp.ones(self.item_count)

    def probe(self, context_state):
        return jnp.ones(self.item_count) * 0.5


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
    model = _FakeModel(recallable=[True, True, False])

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

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
    model = _FakeModel(recallable=[False, False, False])

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

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
    model = _FakeModel(recallable=[True] * 5, recall_total=0)

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

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
        _FakeModel(recallable=[True] * 5, recall_total=0)  # type: ignore[arg-type]
    )
    p_late = policy.stop_probability(
        _FakeModel(recallable=[True] * 5, recall_total=4)  # type: ignore[arg-type]
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
    model = _FakeModel(recallable=[True, True, True], is_active=False)

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

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
    model = _FakeModel(recallable=[True, True, True])

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

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
    model = _FakeModelWithMcf(recallable=[False, False, False])

    # Act / When
    p_stop = policy.stop_probability(model)  # type: ignore[arg-type]

    # Assert / Then
    assert jnp.isclose(p_stop, 1.0).item()
