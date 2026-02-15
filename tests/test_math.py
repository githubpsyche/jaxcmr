import jax.numpy as jnp

from jaxcmr.math import (
    power_scale,
    power_scale_absolute,
    simple_power_scale,
    exponential_primacy_decay,
    normalize_magnitude,
    cosine_similarity_matrix,
)


def test_normalize_magnitude_returns_unit_vector():
    """Behavior: ``normalize_magnitude`` scales a vector to unit length.

    Given:
      - A non-zero vector.
    When:
      - ``normalize_magnitude`` is applied.
    Then:
      - The resulting vector has magnitude approximately 1.
    Why this matters:
      - Context integration relies on unit-length representations.
    """
    # Arrange / Given
    vec = jnp.array([3.0, 4.0])

    # Act / When
    result = normalize_magnitude(vec)

    # Assert / Then
    assert jnp.isclose(jnp.linalg.norm(result), 1.0).item()


def test_normalize_magnitude_handles_zero_vector():
    """Behavior: ``normalize_magnitude`` does not produce NaN for zeros.

    Given:
      - A zero vector.
    When:
      - ``normalize_magnitude`` is applied.
    Then:
      - The result contains no NaN values.
    Why this matters:
      - Prevents NaN propagation in edge-case inputs.
    """
    # Arrange / Given
    vec = jnp.zeros(3)

    # Act / When
    result = normalize_magnitude(vec)

    # Assert / Then
    assert not jnp.any(jnp.isnan(result)).item()


def test_exponential_primacy_decay_returns_scale_plus_one_at_zero():
    """Behavior: ``exponential_primacy_decay`` equals ``scale + 1`` at index 0.

    Given:
      - Study index 0, scale 2.0, decay 0.5.
    When:
      - ``exponential_primacy_decay`` is evaluated.
    Then:
      - The result is ``2.0 * exp(0) + 1 = 3.0``.
    Why this matters:
      - Confirms the primacy formula at the boundary.
    """
    # Arrange / Given
    scale, decay = 2.0, 0.5

    # Act / When
    result = exponential_primacy_decay(jnp.array(0), scale, decay)

    # Assert / Then
    assert jnp.isclose(result, 3.0).item()


def test_exponential_primacy_decay_decreases_with_index():
    """Behavior: ``exponential_primacy_decay`` decreases monotonically.

    Given:
      - Indices 0 through 4.
    When:
      - ``exponential_primacy_decay`` is evaluated for each.
    Then:
      - Each successive value is smaller than the previous.
    Why this matters:
      - Ensures primacy weighting decays as expected.
    """
    # Arrange / Given
    indices = jnp.arange(5)
    scale, decay = 1.0, 0.3

    # Act / When
    values = exponential_primacy_decay(indices, scale, decay)

    # Assert / Then
    assert jnp.all(values[:-1] > values[1:]).item()


def test_power_scale_returns_identity_when_scale_is_one():
    """Behavior: ``power_scale`` is a no-op when scale equals 1.

    Given:
      - A positive vector and scale = 1.
    When:
      - ``power_scale`` is applied.
    Then:
      - The output matches the input.
    Why this matters:
      - Confirms the identity shortcut in the implementation.
    """
    # Arrange / Given
    values = jnp.array([0.1, 0.5, 0.9])

    # Act / When
    result = power_scale(values, jnp.array(1.0))

    # Assert / Then
    assert jnp.allclose(result, values).item()


def test_power_scale_preserves_ordering():
    """Behavior: ``power_scale`` preserves element ordering.

    Given:
      - A sorted positive vector and a non-unity scale.
    When:
      - ``power_scale`` is applied.
    Then:
      - The rank order of elements is preserved.
    Why this matters:
      - Luce choice rule depends on correct ordering after scaling.
    """
    # Arrange / Given
    values = jnp.array([0.1, 0.3, 0.6])

    # Act / When
    result = power_scale(values, jnp.array(2.0))

    # Assert / Then
    assert jnp.all(result[:-1] < result[1:]).item()


def test_power_scale_absolute_identity_when_exponent_one():
    """Behavior: ``power_scale_absolute`` preserves values when exponent is 1.

    Given:
      - A positive vector and exponent = 1.
    When:
      - ``power_scale_absolute`` is applied.
    Then:
      - The output is close to the input.
    Why this matters:
      - Validates the magnitude-preserving identity case.
    """
    # Arrange / Given
    values = jnp.array([0.2, 0.5, 0.8])

    # Act / When
    result = power_scale_absolute(values, 1.0)

    # Assert / Then
    assert jnp.allclose(result, values, atol=1e-5).item()


def test_simple_power_scale_returns_exact_power():
    """Behavior: ``simple_power_scale`` returns ``value ** scale``.

    Given:
      - Values and a scale factor.
    When:
      - ``simple_power_scale`` is applied.
    Then:
      - The result equals the direct power operation.
    Why this matters:
      - Confirms the unstabilized power path.
    """
    # Arrange / Given
    values = jnp.array([2.0, 3.0])
    scale = jnp.array(3.0)

    # Act / When
    result = simple_power_scale(values, scale)

    # Assert / Then
    expected = jnp.array([8.0, 27.0])
    assert jnp.allclose(result, expected).item()


def test_cosine_similarity_orthogonal_vectors():
    """Behavior: ``cosine_similarity_matrix`` returns zero for orthogonal vectors.

    Given:
      - Two orthogonal feature vectors.
    When:
      - The cosine similarity matrix is computed.
    Then:
      - Off-diagonal entries are approximately zero.
    Why this matters:
      - Validates the similarity baseline for unrelated items.
    """
    # Arrange / Given
    features = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    # Act / When
    sim = cosine_similarity_matrix(features)

    # Assert / Then
    assert jnp.isclose(sim[0, 1], 0.0, atol=1e-5).item()
    assert jnp.isclose(sim[1, 0], 0.0, atol=1e-5).item()


def test_cosine_similarity_identical_vectors():
    """Behavior: ``cosine_similarity_matrix`` returns 1.0 on the diagonal.

    Given:
      - Feature vectors with nonzero entries.
    When:
      - The cosine similarity matrix is computed.
    Then:
      - Diagonal entries are approximately 1.0.
    Why this matters:
      - Self-similarity must be maximal.
    """
    # Arrange / Given
    features = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    # Act / When
    sim = cosine_similarity_matrix(features)

    # Assert / Then
    assert jnp.allclose(jnp.diag(sim), jnp.ones(2), atol=1e-5).item()
