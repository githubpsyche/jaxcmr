from jax import lax, jit, numpy as jnp, vmap
from jaxcmr.context import (
    TemporalContext,
    integrate,
    integrate_outlist_context,
)
import pytest

def test_single_size_create():
    TemporalContext.create(5)

def test_erroneous_multi_size_create():

    with pytest.raises(TypeError):
        vmap(TemporalContext.create)(jnp.array([5, 10, 15]))

def test_multi_size_create():
    vmap(TemporalContext.create, in_axes=(0, None))(
        jnp.array([5, 10, 15]), 15)
    
def test_integrate_second_context_unit():
    """
    Test that the second context unit is updated correctly in the integration step,
    whether function is jit compiled or not
    """

    encoding_drift_rate = 0.3
    context = vmap(TemporalContext.create, in_axes=(0, None))(
        jnp.array([5, 10, 15]), 15)
    
    max_item_count = 15
    context_input = jnp.zeros(max_item_count + 2)
    context_input = context_input.at[1].set(1)
    context_input = context_input / jnp.sqrt(jnp.sum(jnp.square(context_input)))

    _integrate = jit(vmap(integrate, in_axes=(0, None, None)))
    
    context = _integrate(context, context_input, encoding_drift_rate)

    desired_result = jnp.array([0.9539392, 0.3])
    assert jnp.allclose(context.state[:, :2], desired_result)

def test_integrate_outlist_context():
    """Test that the outlist context unit is updated correctly in the integration step"""

    context = vmap(TemporalContext.create, in_axes=(0, None))(
        jnp.array([5, 10, 15]), 15)
    drift_rate = 0.5

    context = vmap(integrate_outlist_context, in_axes=(0,None))(context, drift_rate)