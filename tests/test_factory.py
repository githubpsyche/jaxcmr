import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.components.factory import build_trial_connections_from_similarity
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import make_dataset
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
    data = make_dataset([1, 2, 0], pres_itemnos=[1, 2, 3])
    data['valence'] = jnp.array([[0, 1, 0]])
    ids = jnp.array([[[1, 2, 3]], [[3, 2, 1]]])
    inputs = jnp.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
    connections = lambda selected: factory({**data, 'pres_itemids': selected}, inputs).trial_connections
    actual = jax.vmap(connections)(ids)
    expected = jnp.stack([connections(selected) for selected in ids])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual[1, 0], actual[0, 0, ::-1, ::-1])


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
