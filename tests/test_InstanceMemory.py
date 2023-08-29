from jax import numpy as jnp, jit, lax, disable_jit
from jaxcmr.memory import (
    LinearAssociativeMcf,
    InstanceMcf,
    associate,
    probe
)
from jaxcmr.helpers import power_scale
import plum
import pytest


class TestInstanceMemory:
    item_count = 16
    learning_rate = 0.10455606050373444
    shared_support = 0.016122091797498662
    item_support = 0.8877852952105489
    trace_sensitivity = 3.
    choice_sensitivity = 3.0
    contexts = jnp.eye(item_count, item_count + 2, 1)
    items = jnp.eye(item_count, dtype=jnp.float32)

    def test_initialize_mcf(self):
        "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
        linear_mcf = LinearAssociativeMcf.create(
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
        )
        instance_mcf = InstanceMcf.create(
            self.item_count,
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
            1.
        )

        linear_mcf_orthonormal_activations = lax.map(
            lambda context: probe(linear_mcf, context), self.contexts)
        instance_mcf_orthonormal_activations = lax.map(
            lambda context: probe(instance_mcf, context), self.contexts)

        assert jnp.allclose(
            linear_mcf_orthonormal_activations,
            instance_mcf_orthonormal_activations
        )

    def test_associate(self):
        linear_mcf = LinearAssociativeMcf.create(
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
        )
        instance_mcf = InstanceMcf.create(
            self.item_count,
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
            1.
        )

        linear_mcf = associate(linear_mcf, 0.1, self.contexts[0], self.items[0])
        instance_mcf = associate(instance_mcf, 0.1, self.contexts[0], self.items[0])

        linear_mcf_orthonormal_activations = lax.map(
            lambda context: probe(linear_mcf, context), self.contexts)
        instance_mcf_orthonormal_activations = lax.map(
            lambda context: probe(instance_mcf, context), self.contexts)

        assert jnp.allclose(
            linear_mcf_orthonormal_activations,
            instance_mcf_orthonormal_activations
        )

    def test_power_scale(self):
        mcf2 = InstanceMcf.create(
            self.item_count,
            self.item_count,
            self.shared_support,
            self.item_support,
            1.,
            1.
        )

        mcf = InstanceMcf.create(
            self.item_count,
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
            1.
        )

        mcf3 = InstanceMcf.create(
            self.item_count,
            self.item_count,
            self.shared_support,
            self.item_support,
            self.choice_sensitivity,
            self.trace_sensitivity
        )

        activation1 = probe(mcf, self.contexts[0])
        activation2 = probe(mcf2, self.contexts[0])
        activation3 = probe(mcf3, self.contexts[0])

        assert activation3[0] > activation2[0]

        assert jnp.allclose(
            activation1, power_scale(activation2, self.choice_sensitivity)
        )

        assert jnp.allclose(
            activation1 / jnp.sum(activation1),
            jnp.power(activation2, self.choice_sensitivity)
            / jnp.sum(jnp.power(activation2, self.choice_sensitivity)),
        )
