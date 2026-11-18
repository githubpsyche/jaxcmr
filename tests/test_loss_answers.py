"""Independent answers: event probabilities are fixed, so products are analytic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import make_dataset
from jaxcmr.fitting.scipy import make_scipy_loss_fn
from jaxcmr.fitting.parameters import ParameterSpace
from jaxcmr.loss import sequence_likelihood as sequence
from jaxcmr.experimental import set_permutation_likelihood as bag


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
        (sequence.ExcludeFirstRecallLikelihoodLoss, 0.2 * 0.1 * 0.1 * 0.1),
        (sequence.ExcludeTerminationLikelihoodLoss, 0.4 * 0.2 * 0.4),
        (bag.RecallPrefixLikelihoodLoss, 2 * 0.4 * 0.2 * 0.4),
        (bag.RecallSetLikelihoodLoss, 2 * 0.4 * 0.2 * 0.1 * 0.4 * 0.1),
    ],
)
def test_likelihood_known_event_products(loss_cls, product):
    # Recalls are positions: [1,2] on [3,1,2] means items [3,1].
    # The first set has 2 distinct orders; the second has 1.
    # Complete sets include exactly one STOP per trial, regardless of padding.
    data = make_dataset(
        jnp.array([[1, 2, 0], [2, 0, 0]]), pres_itemnos=jnp.array([[3, 1, 2], [2, 3, 1]])
    )
    loss = loss_cls(FixedFactory, data, None)
    answer = jax.jit(lambda: loss(jnp.array([0, 1]), {}, (), jnp.empty((0, 2))))()
    np.testing.assert_allclose(
        answer, [-np.log(product)] * 2, rtol=1e-6,
        err_msg="Event likelihood must match the counted product",
    )


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
    np.testing.assert_allclose(
        function(inputs), [4.0, 20.0], rtol=1e-6,
        err_msg="Parameter names must remain aligned",
    )
    assert function(inputs[:, 0]).shape == ()
    assert function(inputs[:, 0]) == pytest.approx(4.0)


@pytest.mark.parametrize("recalls,orders,items_product", [([1,2,0],2,.2*.3), ([1,1,2],3,.2*.2*.3), ([0,0,0],1,1.)])
@pytest.mark.parametrize("loss_cls,stop", [(bag.RecallSetLikelihoodLoss,.1), (bag.RecallPrefixLikelihoodLoss,1.)])
def test_unordered_probability_counts_distinct_orders_and_one_stop(recalls, orders, items_product, loss_cls, stop):
    data = make_dataset(jnp.array([recalls]), pres_itemnos=jnp.array([[1,2,3]]))
    loss = loss_cls(FixedFactory, data, None, permutation_count=7, seed=19)
    value = loss(jnp.array([0]), {}, (), jnp.empty((0,1)))[0]
    # Every ordering has the same product for this model, so the estimate is exact.
    assert value == pytest.approx(-np.log(orders * items_product * stop), abs=1e-6)


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("loss_cls,stop", [(bag.RecallSetLikelihoodLoss, .1), (bag.RecallPrefixLikelihoodLoss, 1.)])
def test_unordered_sampling_matches_enumerated_order_dependent_probabilities(double, loss_cls, stop):
    class HistoryModel(FixedProbabilities):
        def __init__(self):
            self.last = jnp.array(0)
        def retrieve(self, item):
            return self.replace(last=item)
        def outcome_probability(self, item):
            # P(1,2)=.2*.5, P(2,1)=.3*.4; STOP is .1.
            choices = jnp.array([[.1,.2,.3],[.1,.2,.5],[.1,.4,.3]])
            return choices[self.last, item]
    class HistoryFactory(FixedFactory):
        def create_trial_model(self, trial_index, parameters):
            return HistoryModel()
    with jax.enable_x64(double):
        data = make_dataset(jnp.array([[1,2]]), pres_itemnos=jnp.array([[1,2]]))
        loss = loss_cls(HistoryFactory, data, None, permutation_count=4096, seed=4)
        value = jax.jit(lambda: loss(jnp.array([0]), {}, (), jnp.empty((0,1)))[0])()
        # Enumerate the two distinct sequences independently of the estimator.
        exact = (.2*.5 + .3*.4) * stop
        np.testing.assert_allclose(jnp.exp(-value), exact, rtol=.005)
