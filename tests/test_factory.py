import jax.numpy as jnp
import numpy as np

from jaxcmr.components.factory import build_trial_connections


def test_returns_zero_matrices_when_features_none():
    """Behavior: ``build_trial_connections`` returns zeros without features.

    Given:
      - A presentation list and ``features=None``.
    When:
      - ``build_trial_connections`` is called.
    Then:
      - All connection matrices are zero.
    Why this matters:
      - Models without semantic features must get empty connections.
    """
    # Arrange / Given
    present_lists = np.array([[1, 2, 3]])

    # Act / When
    result = build_trial_connections(present_lists, features=None)

    # Assert / Then
    assert result.shape == (1, 3, 3)
    assert jnp.all(result == 0.0).item()


def test_diagonal_is_zero_with_features():
    """Behavior: ``build_trial_connections`` zeros the diagonal.

    Given:
      - Feature vectors and a single-trial presentation.
    When:
      - ``build_trial_connections`` is called.
    Then:
      - Diagonal entries of the connection matrix are zero.
    Why this matters:
      - Self-connections must be excluded from semantic cueing.
    """
    # Arrange / Given
    features = jnp.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    present_lists = np.array([[1, 2, 3]])

    # Act / When
    result = build_trial_connections(present_lists, features)

    # Assert / Then
    assert jnp.allclose(jnp.diag(result[0]), jnp.zeros(3), atol=1e-6).item()


def test_similar_features_produce_positive_connections():
    """Behavior: ``build_trial_connections`` yields positive off-diagonal for similar items.

    Given:
      - Two nearly identical feature vectors.
    When:
      - ``build_trial_connections`` is called.
    Then:
      - Off-diagonal connection values are positive.
    Why this matters:
      - Semantic similarity should produce retrieval competition.
    """
    # Arrange / Given
    features = jnp.array([[1.0, 0.0], [0.9, 0.1]])
    present_lists = np.array([[1, 2]])

    # Act / When
    result = build_trial_connections(present_lists, features)

    # Assert / Then
    assert float(result[0, 0, 1]) > 0.0
    assert float(result[0, 1, 0]) > 0.0
