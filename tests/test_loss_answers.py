"""Independent answers: event probabilities are fixed, so products are analytic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.helpers import make_dataset
from jaxcmr.fitting.scipy import make_scipy_loss_fn
from jaxcmr.fitting.parameters import ParameterSpace
from jaxcmr.loss import sequence_likelihood as sequence, base_sequence_likelihood as base
from jaxcmr.loss import (
    transform_sequence_likelihood as transform,
    set_permutation_likelihood as bag,
)


class FixedProbabilities(Pytree):
    def __init__(self):
        self.probabilities = jnp.array([0.1, 0.2, 0.3, 0.4])

    def experience(self, item):
        return self

    def start_retrieving(self):
        return self

    def retrieve(self, item):
        return self

    def outcome_probability(self, item):
        return self.probabilities[item]


class FixedFactory:
    def __init__(self, dataset, features):
        pass

    def create_trial_model(self, trial_index, parameters):
        return FixedProbabilities()


@pytest.mark.parametrize(
    "loss_cls,product",
    [
        (sequence.MemorySearchLikelihoodLoss, 0.4 * 0.2 * 0.1 * 0.4 * 0.1 * 0.1),
        (base.ExcludeFirstRecallLikelihoodLoss, 0.2 * 0.1 * 0.1 * 0.1),
        (base.ExcludeTerminationLikelihoodLoss, 0.4 * 0.2 * 0.4),
        (transform.ExcludeFirstRecallLikelihoodLoss, 0.2 * 0.1 * 0.1 * 0.1),
        (transform.ExcludeTerminationLikelihoodLoss, 0.4 * 0.2 * 0.4),
        (bag.ExcludeTerminationLikelihoodLoss, 0.4 * 0.2 * 0.4),
        (bag.IncludeTerminationLikelihoodLoss, 0.4 * 0.2 * 0.1 * 0.4 * 0.1 * 0.1),
    ],
)
def test_likelihood_known_event_products(loss_cls, product):
    # Recalls are positions: [1,2] on [3,1,2] means items [3,1].
    # The second row recalls item 3. Permuting a bag leaves its product unchanged.
    data = make_dataset(
        jnp.array([[1, 2, 0], [2, 0, 0]]), pres_itemnos=jnp.array([[3, 1, 2], [2, 3, 1]])
    )
    loss = loss_cls(FixedFactory, data, None)
    answer = jax.jit(lambda: loss(jnp.array([0, 1]), {}, (), jnp.empty((0, 2))))()
    np.testing.assert_allclose(answer, [-np.log(product)] * 2, rtol=1e-6)
    if loss_cls is bag.IncludeTerminationLikelihoodLoss:
        full = make_dataset(jnp.array([[1, 2, 3]]))
        with pytest.raises(ValueError, match="at least one padding slot"):
            loss_cls(FixedFactory, full, None)


@pytest.mark.parametrize("transformed", [False, True])
def test_scipy_adapter_preserves_scalar_population_order_and_transform(transformed):
    def loss(indices, defaults, names, values):
        parameters = defaults | dict(zip(names, values))
        return (parameters["scale"] - 2) ** 2 + 3 * (parameters["offset"] - 5) ** 2 + indices.sum()

    names = ("offset", "scale")
    space = (
        ParameterSpace({"offset": [1.0, 9.0], "scale": [1.0, 16.0]}, {"scale": "log"})
        if transformed
        else None
    )
    function = make_scipy_loss_fn(
        loss, jnp.array([1, 3]), {"scale": 99.0, "offset": 99.0}, names, space
    )
    # Columns represent (offset=5, scale=2), (offset=3, scale=4).
    inputs = (
        np.array([[0.5, 0.25], [0.25, 0.5]]) if transformed else np.array([[5.0, 3.0], [2.0, 4.0]])
    )
    np.testing.assert_allclose(function(inputs), [4.0, 20.0], rtol=1e-6)
    assert function(inputs[:, 0]).shape == ()
    assert function(inputs[:, 0]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="Expected"):
        function(jnp.ones(3))
