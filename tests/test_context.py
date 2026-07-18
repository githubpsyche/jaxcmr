import jax.numpy as jnp
from jaxcmr.components.context import TemporalContext, init


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
    context = init(2)
    input_vec = jnp.array([0.0, 1.0, 0.0])

    # Act / When
    updated = context.integrate(input_vec, 0.5)

    # Assert / Then
    assert jnp.isclose(jnp.linalg.norm(updated.state), 1.0)


def test_leaves_context_unchanged_when_integrating_itself():
    """Behavior: Treat the current context as a fixed point.

    Given:
      - A unit-length context state.
      - An identical context input.
    When:
      - ``integrate`` is invoked.
    Then:
      - The direction of the context remains unchanged.
    Why this matters:
      - Moving context toward its current state must not rotate it.
    """
    # Arrange / Given
    context = init(1).replace(state=jnp.array([0.6, 0.8]))

    # Act / When
    updated = context.integrate(context.state, 0.5)

    # Assert / Then
    assert jnp.allclose(updated.state, context.state)


def test_matches_scalar_overlap_update_for_nonorthogonal_input():
    """Behavior: Use one scalar overlap to retain the previous context.

    Given:
      - Unit-length, nonorthogonal context and input vectors.
      - A drift rate of ``0.4``.
    When:
      - ``integrate`` is invoked.
    Then:
      - The state matches the hand-calculated CMR context update.
    Why this matters:
      - CMR scales the whole prior context by one ``rho`` value.
    """
    # Arrange / Given
    context = init(2).replace(state=jnp.array([0.6, 0.8, 0.0]))
    input_vec = jnp.array([0.8, 0.0, 0.6])

    # Act / When
    updated = context.integrate(input_vec, 0.4)

    # Assert / Then
    expected = jnp.array([0.7666461, 0.5955281, 0.24])
    assert jnp.allclose(updated.state, expected)


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


def test_outlist_integration_matches_scalar_overlap_update():
    """Behavior: Use scalar overlap when mixing in-list and out-list context.

    Given:
      - An expanded context that overlaps the in-list input.
      - A mixed in-list and out-list input.
    When:
      - ``integrate_with_outlist`` is invoked.
    Then:
      - The state matches the hand-calculated CMR context update.
    Why this matters:
      - Out-list drift must retain the old context as one vector rather than
        scaling its coordinates independently.
    """
    # Arrange / Given
    context = TemporalContext.init_expanded(2).replace(
        state=jnp.array([0.6, 0.8, 0.0, 0.0, 0.0])
    )
    inlist = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])

    # Act / When
    updated = context.integrate_with_outlist(
        inlist,
        ratio=0.6,
        drift_rate=0.4,
    )

    # Assert / Then
    expected = jnp.array([0.4128904, 0.8833407, 0.0, 0.2218801, 0.0])
    assert jnp.allclose(updated.state, expected)

