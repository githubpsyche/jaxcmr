from jax import numpy as jnp, jit
from jaxcmr.memory import (
    LinearAssociativeMcf,
    LinearAssociativeMfc,
    associate,
    probe,
    scale_activation,
)
import plum
import pytest


class TestLinearAssociativeMemory:
    item_count = 16
    learning_rate = 0.10455606050373444
    shared_support = 0.016122091797498662
    item_support = 0.8877852952105489
    choice_sensitivity = 3.0
    contexts = jnp.eye(item_count, item_count + 2, 1)
    items = jnp.eye(item_count, dtype=jnp.float32)

    def test_initialize_mfc(self):
        "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
        memory2 = LinearAssociativeMfc.create(self.item_count, self.learning_rate)
        memory1 = LinearAssociativeMfc.create(self.items, self.learning_rate)
        assert jnp.array_equal(memory1.state, memory2.state)

    def test_initialize_mcf(self):
        "Initializing with 16 or jnp.eye(16) produces equivalent memory states"

        @jit
        def f():
            memory1 = LinearAssociativeMcf.create(
                self.item_count,
                self.shared_support,
                self.item_support,
                self.choice_sensitivity,
            )
            memory2 = LinearAssociativeMcf.create(
                self.items,
                self.shared_support,
                self.item_support,
                self.choice_sensitivity,
            )
            return memory1.state, memory2.state

        assert jnp.array_equal(*f())

    def test_generalized_mfc(self):
        "Initializing with distributed item representations connects a context unit to applicable features."
        items = jnp.eye(self.item_count, self.item_count + 4)
        items = items.at[1:, 0].set(0.2)
        memory = LinearAssociativeMfc.create(items, 0.1)
        desired_result = jnp.array(
            [
                0.0,
                0.9,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.17999999,
                0.0,
            ],
            dtype=jnp.float32,
        )
        assert jnp.allclose(memory.state[0], desired_result)

    def test_dispatch_associate(self):
        "Calls to associate should work for instances of subtypes of LinearAssociativeMemory"
        mfc = LinearAssociativeMfc.create(self.item_count, self.learning_rate)
        mcf = LinearAssociativeMcf.create(
            self.item_count, self.shared_support, self.item_support
        )
        mfc = associate(mfc, 0.1, self.items[0], self.contexts[0])
        mcf = associate(mcf, 0.1, self.contexts[0], self.items[0])

    def test_shape_selective_dispatch(self):
        "Associating with wrong-shaped inputs should raise NotFoundLookupError at dispatch"

        memory = LinearAssociativeMfc.create(self.item_count, self.learning_rate)
        with pytest.raises(plum.NotFoundLookupError):
            associate(memory, self.items[0], self.items[0])

    def test_probe_result_length(self):
        "Probe results should have length equal to the number of items plus two"
        mfc = LinearAssociativeMfc.create(self.item_count, self.learning_rate)
        assert len(probe(mfc, self.items[0])) == self.item_count + 2

    def test_probe_result_magnitude(self):
        "Probe results should be normalized to have a magnitude of 1."
        mfc = LinearAssociativeMfc.create(self.item_count, self.learning_rate)
        mfc = associate(mfc, 0.1, self.items[0], self.contexts[0])
        probe_result = probe(mfc, self.items[0])
        assert jnp.allclose(jnp.linalg.norm(probe_result), 1.0)

    def test_choice_sensitivity(self):
        mcf = LinearAssociativeMcf.create(
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
        )
        mcf2 = LinearAssociativeMcf.create(self.items, self.shared_support, self.item_support, 1.0)

        activation1 = probe(mcf, self.contexts[0])
        activation2 = probe(mcf2, self.contexts[0])
        
        assert jnp.allclose(activation1, scale_activation(activation2, self.choice_sensitivity))
        assert jnp.allclose(
            activation1 / jnp.sum(activation1),
            jnp.power(activation2, self.choice_sensitivity)
            / jnp.sum(jnp.power(activation2, self.choice_sensitivity)),
        )
