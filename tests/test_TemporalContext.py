from jaxcmr.context import initialize_temporal_context, integrate, integrate_delay_context, state
from jax import jit, numpy as jnp


def test_integrate_second_context_unit():

    @jit
    def f():
        item_count = 10
        encoding_drift_rate = 0.3
        context = initialize_temporal_context(item_count)

        context_input = jnp.zeros(item_count + 2)
        context_input = context_input.at[1].set(1)
        context_input = context_input / jnp.sqrt(jnp.sum(jnp.square(context_input)))

        return integrate(context, context_input, encoding_drift_rate)

    desired_result = jnp.array([0.9539392, 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert jnp.allclose(state(f()), desired_result)


def test_integrate_delay_context():

    @jit
    def f():
        item_count = 5
        context = initialize_temporal_context(item_count)
        drift_rate = 0.5
        return integrate_delay_context(context, drift_rate)
    
    expected_state = jnp.array([0.8660254, 0., 0., 0., 0., 0., 0.5])
    assert jnp.allclose(state(f()), expected_state)