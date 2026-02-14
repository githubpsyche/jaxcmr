import jax.numpy as jnp
from jaxcmr.components.context import init as init_context
from jaxcmr.components.linear_memory import LinearMemory, init_mfc, init_mcf


def test_updates_state_when_associating_patterns():
    """Behavior: Learning adds the outer product to memory state.

    Given:
      - A memory with zero weights.
      - Input and output patterns.
    When:
      - ``associate`` is invoked.
    Then:
      - State equals the outer product scaled by the learning rate.
    Why this matters:
      - Confirms learning rule implementation.
    """
    # Arrange / Given
    mem = LinearMemory(jnp.zeros((2, 2)))
    in_pattern = jnp.array([1.0, 0.0])
    out_pattern = jnp.array([0.5, 0.5])
    lr = 0.3

    # Act / When
    updated = mem.associate(in_pattern, out_pattern, lr)

    # Assert / Then
    expected = mem.state + lr * jnp.outer(in_pattern, out_pattern)
    assert jnp.allclose(updated.state, expected)


def test_returns_weighted_output_when_probing():
    """Behavior: Probing retrieves the weighted sum of associations.

    Given:
      - A memory with known weights.
      - An input pattern.
    When:
      - ``probe`` is called.
    Then:
      - Output matches matrix multiplication.
    Why this matters:
      - Verifies retrieval accuracy.
    """
    # Arrange / Given
    state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mem = LinearMemory(state)
    in_pattern = jnp.array([1.0, 0.0])

    # Act / When
    result = mem.probe(in_pattern)

    # Assert / Then
    assert jnp.allclose(result, jnp.array([1.0, 2.0]))


def test_clears_state_row_when_zeroing_index():
    """Behavior: ``zero_out`` removes associations for an index.

    Given:
      - A memory with nonzero weights.
    When:
      - ``zero_out`` is called.
    Then:
      - The targeted row becomes zero.
    Why this matters:
      - Enables selective forgetting.
    """
    # Arrange / Given
    state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mem = LinearMemory(state)

    # Act / When
    updated = mem.zero_out(1)

    # Assert / Then
    assert jnp.allclose(updated.state, jnp.array([[1.0, 2.0], [0.0, 0.0]]))


def test_initializes_mfc_with_shifted_identity_when_init_mfc():
    """Behavior: ``init_mfc`` seeds item-to-context associations on a superdiagonal.

    Given:
      - A list length, learning rate, and context with 5 features.
    When:
      - ``init_mfc`` is invoked.
    Then:
      - Resulting matrix has ``1 - learning_rate`` on the superdiagonal.
    Why this matters:
      - Establishes baseline item-context links.
    """
    # Arrange / Given
    list_length = 3
    lr = 0.2
    context = init_context(4)  # size = 5

    # Act / When
    mem = init_mfc(list_length, {"learning_rate": lr}, context)

    # Assert / Then
    expected = jnp.eye(list_length, context.size, 1) * (1 - lr)
    assert jnp.allclose(mem.state, expected)


def test_initializes_mcf_with_shared_and_item_support_when_init_mcf():
    """Behavior: ``init_mcf`` mixes shared and item-specific support.

    Given:
      - A list length, support parameters, and context with 4 features.
    When:
      - ``init_mcf`` is called.
    Then:
      - Context rows contain shared support with extra item support on the diagonal.
    Why this matters:
      - Provides expected starting weights for retrieval.
    """
    # Arrange / Given
    list_length = 2
    item_support = 0.7
    shared_support = 0.3
    context = init_context(3)  # size = 4

    # Act / When
    params = {"item_support": item_support, "shared_support": shared_support}
    mem = init_mcf(list_length, params, context)

    # Assert / Then
    expected = jnp.array([
        [0.0, 0.0],
        [item_support, shared_support],
        [shared_support, item_support],
        [shared_support, shared_support],
    ])
    assert jnp.allclose(mem.state, expected)
