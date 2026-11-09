"""Shape, ordering and precision contracts for shared fitting coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.fitting.parameters import ParameterSpace


@pytest.mark.parametrize("method", ["to_physical", "to_unit"])
@pytest.mark.parametrize(
    "shape,compiled", [((), False), ((1,), False), ((4, 3), False), ((3,), True)]
)
def test_parameter_conversion_rejects_wrong_width(method, shape, compiled):
    """A missing or extra coordinate must not broadcast or disappear under JIT."""
    space = ParameterSpace({"a": [0.0, 10.0], "b": [100.0, 200.0]})
    convert = getattr(space, method)
    if compiled:
        convert = jax.jit(convert)
    with pytest.raises(ValueError, match="final axis of 2 parameters"):
        convert(jnp.ones(shape))


def test_parameter_conversion_preserves_batch_axes_and_name_order():
    """Round trips and gradients use the final axis, including multiple batch axes."""
    with jax.enable_x64(True):
        space = ParameterSpace(
            {"retention_decay": [0.01, 10.0], "amplitude": [0.1, 100.0], "drift": [0.1, 0.9]},
            {"retention_decay": "exp_neg", "amplitude": "log"},
        )
        unit = jnp.linspace(0.1, 0.9, 24).reshape(2, 4, 3)
        physical = jax.jit(space.to_physical)(unit)
        np.testing.assert_allclose(jax.jit(space.to_unit)(physical), unit, atol=1e-12)
        np.testing.assert_allclose(
            jax.vmap(jax.vmap(space.to_physical))(unit), physical, atol=1e-12
        )
        derivative = jax.jit(jax.grad(lambda x: space.to_physical(x).sum()))(unit)
        assert derivative.shape == unit.shape
        assert np.all(np.isfinite(derivative))
        assert np.all(derivative[..., 0] < 0)
        assert np.all(derivative[..., 1:] > 0)
