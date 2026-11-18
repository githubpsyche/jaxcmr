import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components.factory import build_trial_connections, build_trial_connections_from_similarity
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.dataset import make_dataset
from jaxcmr.models import (
    additive_semantic_cmr, additive_semantic_matrix_cmr, multiplicative_semantic_cmr,
    additive_semantic_ecmr, additive_semantic_retrieval_bias_ecmr,
)


@pytest.mark.parametrize("module", [
    additive_semantic_cmr, additive_semantic_matrix_cmr, multiplicative_semantic_cmr,
    additive_semantic_ecmr, additive_semantic_retrieval_bias_ecmr,
])
def test_semantic_factories_batch_item_identities(module):
    factory = module.make_factory(linear_memory.init_mfc, linear_memory.init_mcf,
                                  context.init, NoStopTermination)
    # Repeated code 2, an internal padding slot, and a storage width larger
    # than the model's item count all distinguish positions from item codes.
    data = make_dataset([1, 2, 0], pres_itemnos=[2, 1, 2, 0, 3, 0])
    data["listLength"] = jnp.array([[4]])
    data["valence"] = jnp.zeros_like(data["pres_itemnos"])
    ids = jnp.array([[[2, 1, 2, 0, 3, 0]], [[2, 3, 2, 0, 1, 0]]])
    inputs = jnp.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
    builder = (
        build_trial_connections
        if module in (additive_semantic_cmr, multiplicative_semantic_cmr)
        else build_trial_connections_from_similarity
    )

    def connections(selected):
        return factory({**data, "pres_itemids": selected}, inputs).trial_connections

    actual = jax.vmap(connections)(ids)
    expected = builder(jnp.array([[1, 2, 3, 0], [3, 2, 1, 0]]), inputs)
    np.testing.assert_allclose(actual[:, 0], expected)


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


def test_feature_connections_preserve_direction_at_small_magnitudes():
    features = jnp.array([[1e-9, 0], [1, 0], [-1, 0], [0, 0]])
    # IDs 1 and 2 are identical directions. Opposite and zero directions
    # receive the existing positive floor; item self-connections and padding stay zero.
    connections = build_trial_connections(jnp.array([[1, 2, 3, 4, 0]]), features)
    expected = np.full((5, 5), 2**-23, dtype=np.float32)
    np.fill_diagonal(expected, 0)
    expected[0, 1] = expected[1, 0] = 1
    expected[4, :] = expected[:, 4] = 0
    np.testing.assert_array_equal(connections[0], expected)
