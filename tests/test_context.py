"""
# test_context.py

Tests that the `TemporalContext` class and other modules implementing the `Context` interface specified in `jaxcmr/typing.py` are working correctly. 
"""

from jaxcmr.context import TemporalContext
import jax.numpy as jnp

def test_temporal_context():
    "Test that TemporalContext initializes correctly and integrates input."

    # Set up the test parameters
    drift_rate = 0.3
    item_count = 10
    size = item_count + 2
    context = TemporalContext(item_count, size)

    # initial state should be 1.0 at the first element, and 0.0 elsewhere
    assert context.state[0] == 1.0
    assert jnp.all(context.state[1:] == 0.0)

    context_input = jnp.zeros(size).at[-1].set(1)
    new_context = context.integrate(context_input, drift_rate)

    # last element is now non-zero; rest are still 0.0, except for the first element
    assert new_context.state[-1] > 0.0 
    assert jnp.all(new_context.state[1:-1] == 0.0)
    assert new_context.state[0] > 0.0

    # final state vector is unit length
    assert jnp.isclose(jnp.linalg.norm(new_context.state), 1.0, atol=1e-6)

    # test that the initial state is preserved
    assert jnp.all(new_context.initial_state == context.state)
