"""Known values and derivatives at numerical boundaries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr.math import cosine_similarity_matrix, normalize_magnitude, relative_power_scale


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


@pytest.mark.parametrize("sensitivity", (1.0, 2.0, 99.0))
def test_relative_power_gradients_ignore_structural_zeros_and_remain_correct_at_one(sensitivity):
    values = jnp.array([1.0, 0.5, 0.0])
    fn = jax.jit(lambda sensitivity: relative_power_scale(values, sensitivity).sum())
    value, gradient = jax.value_and_grad(fn)(sensitivity)
    np.testing.assert_allclose(value, 1 + 0.5**sensitivity)
    np.testing.assert_allclose(gradient, 0.5**sensitivity * np.log(0.5), atol=1e-14)
    zeros = jnp.zeros(3)
    np.testing.assert_array_equal(relative_power_scale(zeros, 2.0), zeros)
    np.testing.assert_array_equal(jax.jacrev(relative_power_scale)(zeros, 2.0), np.zeros((3, 3)))


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_zero_norm_has_finite_explicit_zero_tangent_under_batched_reverse_mode(scale):
    values = scale * jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    result = jax.jit(jax.vmap(normalize_magnitude))(values)
    np.testing.assert_allclose(result, [[0, 0, 0], [0.6, 0.8, 0]])
    derivative = jax.jit(jax.jacrev(jax.vmap(normalize_magnitude)))(values)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(derivative[0], np.zeros((3, 2, 3)))


def test_normalization_does_not_hide_nonfinite_inputs():
    assert np.all(np.isnan(normalize_magnitude(jnp.array([np.nan, 1.0]))))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cosine_similarity_preserves_directions_and_zero_row_derivatives(dtype):
    features = jnp.array([[3e-9, 4e-9], [3, 4], [-3, -4], [4, -3], [0, 0]], dtype=dtype)
    # The first two rows have the same direction; the next is opposite,
    # the fourth orthogonal, and the last has the explicit zero convention.
    expected = [[1, 1, -1, 0, 0], [1, 1, -1, 0, 0], [-1, -1, 1, 0, 0],
                [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    compiled = jax.jit(cosine_similarity_matrix)
    np.testing.assert_allclose(compiled(features), expected, atol=2e-7)
    derivative = jax.jit(jax.jacrev(cosine_similarity_matrix))(features)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(derivative[4], 0)
    np.testing.assert_array_equal(derivative[:, 4], 0)
    # cos([x, 4], [0, 1]) = 4 / sqrt(x**2 + 16), with derivative -12/125 at x=3.
    gradient = jax.jit(jax.grad(lambda x: cosine_similarity_matrix(
        jnp.array([[x, 4], [0, 1]], dtype=dtype)
    )[0, 1]))(jnp.asarray(3, dtype=dtype))
    np.testing.assert_allclose(gradient, -12 / 125, rtol=2e-6)
