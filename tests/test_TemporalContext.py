"""
Tests for TemporalContext type and functions
"""

# %% Setup

from jaxcmr.context import (
    TemporalContext,
    integrate,
    integrate_outlist_context,
)
from jax import jit, numpy as jnp

def test_integrate_second_context_unit():
    """
    Test that the second context unit is updated correctly in the integration step,
    whether function is jit compiled or not
    """

    def f():
    
        item_count = 10
        encoding_drift_rate = 0.3
        context = TemporalContext.create(item_count)

        context_input = jnp.zeros(item_count + 2)
        context_input = context_input.at[1].set(1)
        context_input = context_input / jnp.sqrt(jnp.sum(jnp.square(context_input)))

        return integrate(context, context_input, encoding_drift_rate)

    desired_result = jnp.array(
        [0.9539392, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert jnp.allclose(jit(f)().state, desired_result)
    assert jnp.allclose(f().state, desired_result)


def test_integrate_outlist_context():
    """Test that the outlist context unit is updated correctly in the integration step"""
    @jit
    def f():
        item_count = 5
        context = TemporalContext.create(item_count)
        drift_rate = 0.5
        return integrate_outlist_context(context, drift_rate)

    expected_state = jnp.array([0.8660254, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    assert jnp.allclose(f().state, expected_state)

#TODO: add test confirming that magnitude of integrated context vector is reliably enforced 1, even after many iterations