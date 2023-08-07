from jax import numpy as jnp
from jaxcmr.memory import LinearAssociativeMemory as lam
import plum
import pytest
from jaxtyping import PyTree

def test_initialize_mfc():
    "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory2 = lam.init_linear_mfc(item_count, learning_rate)
    memory1 = lam.init_linear_mfc(items, learning_rate)
    assert jnp.array_equal(memory1.state, memory2.state)


def test_initialize_mcf():
    "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
    item_count = 16
    shared_support = 0.1
    item_support = 0.9
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory1 = lam.init_linear_mcf(item_count, shared_support, item_support)
    memory2 = lam.init_linear_mcf(items, shared_support, item_support)
    assert jnp.array_equal(memory1.state, memory2.state)


def test_generalized_mfc():
    "Initializing with distributed item representations connects a context unit to applicable features."
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, item_count+4)
    items = items.at[1:, 0].set(.2)
    memory = lam.init_linear_mfc(items, learning_rate)
    desired_result = jnp.array([0.        , 0.9       , 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.17999999, 0.17999999, 0.17999999,
    0.17999999, 0.17999999, 0.        ], dtype=jnp.float32)
    assert jnp.allclose(memory.state[0], desired_result)


def test_dispatch_associate():
    "Calls to associate should work for instances of subtypes of LinearAssociativeMemory"
    item_count = 16
    learning_rate = 0.1
    shared_support = .1
    item_support = .9
    items = jnp.eye(item_count, dtype=jnp.float32)
    contexts = jnp.eye(item_count, item_count+2, 1)
    mfc = lam.init_linear_mfc(item_count, learning_rate)
    mcf = lam.init_linear_mcf(item_count, shared_support, item_support)
    mfc = lam.associate(mfc, .1, items[0], contexts[0])
    mcf = lam.associate(mcf, .1, contexts[0], items[0])


def test_shape_selective_dispatch():
    "Associating with wrong-shaped inputs should raise NotFoundLookupError at dispatch, not execution"
    item_count = 16
    learning_rate = 0.1
    items = jnp.eye(item_count, dtype=jnp.float32)
    memory = lam.init_linear_mfc(item_count, learning_rate)
    with pytest.raises(plum.NotFoundLookupError):
        lam.associate(memory, items[0], items[0])
