import jax.numpy as jnp
from jaxcmr.models.context import TemporalContext


def test_preserves_unit_length_when_integrating_input():
    """Behavior: Maintain unit length after drift integration.

    Given:
      - A context model with two items.
      - A non-zero input vector.
    When:
      - ``integrate`` is invoked.
    Then:
      - The resulting state has magnitude one.
    Why this matters:
      - Prevents drift from altering context magnitude.
    """
    # Arrange / Given
    context = TemporalContext.init(2)
    input_vec = jnp.array([0.0, 1.0, 0.0])

    # Act / When
    updated = context.integrate(input_vec, 0.5)

    # Assert / Then
    assert jnp.isclose(jnp.linalg.norm(updated.state), 1.0)


def test_returns_one_hot_vector_when_requesting_outlist_input():
    """Behavior: Produce one-hot out-of-list input vector.

    Given:
      - A context model expanded for out-of-list units.
    When:
      - ``outlist_input`` is accessed.
    Then:
      - The returned vector has one at the out-of-list index.
    Why this matters:
      - Ensures out-of-list context is uniquely identified.
    """
    # Arrange / Given
    context = TemporalContext.init_expanded(1)

    # Act / When
    vec = context.outlist_input

    # Assert / Then
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.array_equal(vec, expected).item()


def test_advances_outlist_index_when_integrating_with_outlist():
    """Behavior: Increment out-of-list unit after integration.

    Given:
      - A context model with capacity for out-of-list units.
      - An in-list input vector.
    When:
      - ``integrate_with_outlist`` is called.
    Then:
      - ``next_outlist_unit`` increases by one.
    Why this matters:
      - Supports sequential post-study drift representations.
    """
    # Arrange / Given
    context = TemporalContext.init_expanded(2)
    inlist = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])

    # Act / When
    updated = context.integrate_with_outlist(inlist, ratio=0.5, drift_rate=0.5)

    # Assert / Then
    assert updated.next_outlist_unit == context.next_outlist_unit + 1
    assert jnp.isclose(jnp.linalg.norm(updated.state), 1.0)


def test_retains_state_when_drift_rate_zero():
    """Behavior: Leave state unchanged when drift is zero.

    Given:
      - A context model with an initial state.
      - An input vector.
    When:
      - ``integrate`` is called with ``0`` drift rate.
    Then:
      - The state remains identical to the initial state.
    Why this matters:
      - Validates the zero-drift boundary condition.
    """
    # Arrange / Given
    context = TemporalContext.init(2)
    input_vec = jnp.array([0.0, 1.0, 0.0])

    # Act / When
    updated = context.integrate(input_vec, 0.0)

    # Assert / Then
    assert jnp.allclose(updated.state, context.state)
