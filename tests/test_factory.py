import jax.numpy as jnp
import numpy as np
from jaxcmr.components.factory import build_trial_connections_from_similarity


def test_similarity_matrix_indexes_direct_item_ids():
    """direct similarity matrices are sliced with ``pres_itemids - 1``."""
    similarity_matrix = jnp.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    present_lists = np.array([[3, 1]])
    result = build_trial_connections_from_similarity(present_lists, similarity_matrix)
    assert result.shape == (1, 2, 2)
    assert jnp.allclose(result[0], jnp.array([[0.0, 0.3], [0.3, 0.0]])).item()


def test_similarity_matrix_zeros_diagonal_and_padding():
    """direct similarity matrices zero self-connections and padding."""
    similarity_matrix = jnp.array([[jnp.nan, 0.6, 0.2], [0.6, jnp.nan, 0.4], [0.2, 0.4, jnp.nan]])
    present_lists = np.array([[2, 0, 1]])
    result = build_trial_connections_from_similarity(present_lists, similarity_matrix)
    expected = jnp.array([[0.0, 0.0, 0.6], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    assert jnp.allclose(result[0], expected, equal_nan=False).item()
