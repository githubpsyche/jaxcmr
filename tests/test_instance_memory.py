import jax.numpy as jnp
from jaxcmr.components.context import init as init_context
from jaxcmr.components.instance_memory import init_mfc, init_mcf


def test_init_mfc_creates_correct_shape():
    """Behavior: ``init_mfc`` allocates traces with correct dimensions.

    Given:
      - A list length and context with known feature count.
    When:
      - ``init_mfc`` is called.
    Then:
      - State has shape ``(2 * list_length, list_length + context_size)``.
    Why this matters:
      - Pre-allocation must accommodate both pre-experimental and study traces.
    """
    # Arrange / Given
    list_length = 3
    context = init_context(3)  # size = 4
    params = {"learning_rate": 0.2}

    # Act / When
    mem = init_mfc(list_length, params, context)

    # Assert / Then
    expected_rows = 2 * list_length  # list_length pre-experimental + list_length study
    expected_cols = list_length + context.size  # item features + context features
    assert mem.state.shape == (expected_rows, expected_cols)
    assert mem.input_size == list_length
    assert mem.output_size == context.size


def test_init_mcf_creates_correct_shape():
    """Behavior: ``init_mcf`` allocates traces with correct dimensions.

    Given:
      - A list length, context, and support parameters.
    When:
      - ``init_mcf`` is called.
    Then:
      - State has shape ``(2 * list_length, context_size + list_length)``.
    Why this matters:
      - Context-to-item memory must match context and item dimensions.
    """
    # Arrange / Given
    list_length = 3
    context = init_context(3)  # size = 4
    params = {
        "item_support": 0.7,
        "shared_support": 0.3,
        "mcf_trace_sensitivity": 1.0,
    }

    # Act / When
    mem = init_mcf(list_length, params, context)

    # Assert / Then
    expected_rows = 2 * list_length
    expected_cols = context.size + list_length
    assert mem.state.shape == (expected_rows, expected_cols)
    assert mem.input_size == context.size
    assert mem.output_size == list_length


def test_associate_increments_study_index():
    """Behavior: ``associate`` advances the trace write pointer.

    Given:
      - An instance memory with a known study index.
    When:
      - ``associate`` is called with input and output patterns.
    Then:
      - ``study_index`` increments by one.
    Why this matters:
      - Each study event must write to a unique trace slot.
    """
    # Arrange / Given
    list_length = 2
    context = init_context(2)  # size = 3
    params = {"learning_rate": 0.5}
    mem = init_mfc(list_length, params, context)
    original_index = int(mem.study_index)

    in_pattern = jnp.array([1.0, 0.0])
    out_pattern = jnp.array([0.0, 0.5, 0.5])

    # Act / When
    updated = mem.associate(in_pattern, out_pattern, 0.5)

    # Assert / Then
    assert int(updated.study_index) == original_index + 1


def test_probe_recovers_associated_output():
    """Behavior: Probing with a stored input recovers the associated output.

    Given:
      - An instance memory with a known pre-experimental association.
    When:
      - ``probe`` is called with the item's input pattern.
    Then:
      - The output is nonzero in the context feature dimensions.
    Why this matters:
      - Retrieval must recover associations stored during study.
    """
    # Arrange / Given
    list_length = 2
    context = init_context(2)  # size = 3
    params = {"learning_rate": 0.2}
    mem = init_mfc(list_length, params, context)

    # Item 0's input pattern is a one-hot over item features
    item_input = jnp.array([1.0, 0.0])

    # Act / When
    output = mem.probe(item_input)

    # Assert / Then
    assert output.shape == (context.size,)
    assert jnp.any(output != 0.0).item()
