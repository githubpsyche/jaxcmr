"""Behavior required from the bundled simple-pytree implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from jaxcmr._vendor.simple_pytree import Pytree, dataclass, static_field


class State(Pytree):
    label = static_field()

    def __init__(self, values, label="example"):
        self.values = values
        self.label = label


@dataclass
class Record(Pytree):
    values: jax.Array
    label: str = static_field(default="record")


def test_tree_round_trip_preserves_arrays_static_fields_and_key_paths():
    state = State(jnp.array([1.0, 2.0]))
    leaves, structure = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert isinstance(restored, State)
    assert restored.label == "example"
    assert len(leaves) == 1
    np.testing.assert_array_equal(restored.values, state.values)
    keyed_leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    assert keyed_leaves[0][0] == (jax.tree_util.GetAttrKey("values"),)


@pytest.mark.parametrize("state_type", [State, Record])
def test_replace_returns_independent_state_and_rejects_mutation(state_type):
    original = state_type(jnp.array([1.0, 2.0]))
    updated = original.replace(values=jnp.array([3.0, 4.0]))
    assert updated is not original
    assert updated.label == original.label
    np.testing.assert_array_equal(original.values, [1.0, 2.0])
    np.testing.assert_array_equal(updated.values, [3.0, 4.0])
    with pytest.raises(AttributeError, match="immutable"):
        original.values = jnp.zeros(2)
    with pytest.raises(AttributeError, match="Cannot add new fields"):
        original.unexpected = 1


def test_jit_vmap_and_grad_preserve_state_and_static_metadata():
    state = State(jnp.array([1.0, 2.0]))

    @jax.jit
    def scale(value, factor):
        return value.replace(values=value.values * factor)

    batched = jax.vmap(scale, in_axes=(None, 0))(state, jnp.array([2.0, 3.0]))
    assert isinstance(batched, State)
    assert batched.label == state.label
    np.testing.assert_allclose(batched.values, [[2.0, 4.0], [3.0, 6.0]])
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(value.values**2)))(state)
    assert isinstance(gradient, State)
    assert gradient.label == state.label
    np.testing.assert_allclose(gradient.values, [2.0, 4.0])


def test_flax_serialization_round_trip_preserves_static_metadata():
    state = State(jnp.array([1.0, 2.0]), label="saved")
    encoded = serialization.to_state_dict(state)
    assert set(encoded) == {"values"}
    restored = serialization.from_state_dict(state.replace(values=jnp.zeros(2)), encoded)
    assert isinstance(restored, State)
    assert restored.label == "saved"
    np.testing.assert_array_equal(restored.values, state.values)
