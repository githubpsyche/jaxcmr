"""Numerical transforms, ordering and precision of fitting coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxcmr.fitting.parameters import ParameterSpace, bound_params


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
        # Interpolate retention in exp(-decay), amplitude in log space,
        # and drift linearly; these equations do not call ParameterSpace.
        u = np.asarray(unit)
        span = np.exp(-.01) - np.exp(-10.)
        retention = np.exp(-10.) + u[..., 0] * span
        amplitude = .1 * 1000 ** u[..., 1]
        expected = np.stack((-np.log(retention), amplitude, .1 + .8 * u[..., 2]), axis=-1)
        expected_derivative = np.stack(
            (-span / retention, amplitude * np.log(1000), np.full_like(retention, .8)), axis=-1
        )
        np.testing.assert_allclose(physical, expected, atol=1e-12)
        np.testing.assert_allclose(derivative, expected_derivative, atol=1e-12,
                                   err_msg='Transform derivatives must match their equations')


def test_log1p_roundtrip_and_derivative_near_zero():
    with jax.enable_x64(True):
        space = ParameterSpace({"amplitude": [0.0, 100.0]}, {"amplitude": "log1p"})
        physical = jnp.array([[0.0], [1e-08], [1.0], [100.0]])
        unit = jax.jit(space.to_unit)(physical)
        np.testing.assert_allclose(jax.jit(space.to_physical)(unit), physical, rtol=1e-13, atol=1e-14)
        derivative = jax.jit(jax.jacfwd(space.to_physical))(unit[1])
        np.testing.assert_allclose(derivative[0, 0], (1 + 1e-08) * np.log1p(100), rtol=1e-13)


def test_bound_params_clips_each_column_without_changing_inputs():
    params = {"p": jnp.array([-.5, .4, 2.]), "q": jnp.array([3., 5., 8.])}
    bounds = {"p": [0., 1.], "q": [4., 6.]}
    clipped = jax.jit(bound_params)(params, bounds)
    np.testing.assert_allclose(clipped["p"], [0., .4, 1.])
    np.testing.assert_allclose(clipped["q"], [4., 5., 6.])
    np.testing.assert_allclose(params["p"], [-.5, .4, 2.])
    np.testing.assert_allclose(params["q"], [3., 5., 8.])
    assert bounds == {"p": [0., 1.], "q": [4., 6.]}
