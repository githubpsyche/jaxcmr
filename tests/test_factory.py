import jax.numpy as jnp
import numpy as np

from jaxcmr.components.factory import (
    build_trial_connections,
    build_trial_connections_from_similarity,
)


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


def test_similarity_matrix_indexes_direct_item_ids():
    """Behavior: direct similarity matrices are sliced with ``pres_itemids - 1``.

    Given:
      - A wordpool-wide similarity matrix.
      - A study list with non-consecutive item IDs.
    When:
      - ``build_trial_connections_from_similarity`` is called.
    Then:
      - The trial block uses direct item-ID indexing.
    Why this matters:
      - Dupertuys image IDs are direct 1-indexed rows/columns in the
        semantic-rating matrix.
    """
    # Arrange / Given
    similarity_matrix = jnp.array([
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])
    present_lists = np.array([[3, 1]])

    # Act / When
    result = build_trial_connections_from_similarity(present_lists, similarity_matrix)

    # Assert / Then
    assert result.shape == (1, 2, 2)
    assert jnp.allclose(result[0], jnp.array([[0.0, 0.3], [0.3, 0.0]])).item()


def test_similarity_matrix_zeros_diagonal_and_padding():
    """Behavior: direct similarity matrices zero self-connections and padding.

    Given:
      - A similarity matrix with NaNs on the global diagonal.
      - A padded study list.
    When:
      - ``build_trial_connections_from_similarity`` is called.
    Then:
      - Diagonal and padded entries are zero.
    Why this matters:
      - The Dupertuys semantic matrix has NaNs for unrated pairs, while padded
        presentation positions must not contribute semantic support.
    """
    # Arrange / Given
    similarity_matrix = jnp.array([
        [jnp.nan, 0.6, 0.2],
        [0.6, jnp.nan, 0.4],
        [0.2, 0.4, jnp.nan],
    ])
    present_lists = np.array([[2, 0, 1]])

    # Act / When
    result = build_trial_connections_from_similarity(present_lists, similarity_matrix)

    # Assert / Then
    expected = jnp.array([
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0],
    ])
    assert jnp.allclose(result[0], expected, equal_nan=False).item()
