import jax.numpy as jnp
from jaxcmr.models import LinearAssociativeMemory
import pytest
import plum

def test_initialize_mfc():
    "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory2 = LinearAssociativeMemory.initialize_mfc(item_count, learning_rate)
    memory1 = LinearAssociativeMemory.initialize_mfc(items, learning_rate)

    assert jnp.array_equal(memory1.state, memory2.state)


def test_initialize_mcf():
    "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
    item_count = 16
    shared_support = 0.1
    item_support = 0.9
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory1 = LinearAssociativeMemory.initialize_mcf(item_count, shared_support, item_support)
    memory2 = LinearAssociativeMemory.initialize_mcf(items, shared_support, item_support)

    assert jnp.array_equal(memory1.state, memory2.state)


def test_generalized_mfc():
    "Initializing with distributed item representations connects a context unit to applicable features."
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, item_count+4)
    items = items.at[1:, 0].set(.2)

    memory = LinearAssociativeMemory.initialize_mfc(items, learning_rate)

    desired_result = jnp.array([0.        , 0.9       , 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.        ], dtype=jnp.float32)
    assert jnp.allclose(memory.state[0], desired_result)


def test_shape_selective_dispatch():
    "Associating with wrong-shaped inputs should raise NotFoundLookupError at dispatch, not execution"
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory = LinearAssociativeMemory.initialize_mfc(item_count, learning_rate)

    with pytest.raises(plum.NotFoundLookupError):
        LinearAssociativeMemory.associate(memory, items[0], items[0])
        
    