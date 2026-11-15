"""Preparation requirements for the optimized regular-CMR loss."""

from support import OPTIMIZED_CMR_PARAMS
import jax
import jax.numpy as jnp
import pytest
from jaxcmr.helpers import make_dataset
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
from jaxcmr.experimental.optimized_cmr import CMR, ModelFactory as OptimizedFactory, _sequence_likelihood

@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def repeated_trial():
    return make_dataset(
        jnp.array([[31, 29] + [0] * 38]),
        pres_itemnos=jnp.repeat(jnp.arange(1, 21), 2)[None, :],
        listLength=40,
        subject=2,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pres_itemnos", jnp.zeros((1, 40), dtype=int)),
        ("listLength", jnp.array([[39]])),
        ("recalls", jnp.array([[31, 0, 29] + [0] * 37])),
        ("recalls", jnp.array([[31, 32] + [0] * 38])),
        ("recalls", jnp.array([[-1] + [0] * 39])),
    ],
)
def test_unsupported_observations_fail_explicitly(field, value):
    with pytest.raises(ValueError):
        FastCMRLikelihoodLoss(OptimizedFactory, {**repeated_trial(), field: value}, None)


def test_optimized_model_rejects_unsupported_features_during_preparation():
    with pytest.raises(ValueError, match="features must be None"):
        FastCMRLikelihoodLoss(OptimizedFactory, repeated_trial(), jnp.eye(40))


@pytest.mark.parametrize("name,value", [
    ("learn_after_context_update", True), ("learn_after_context_update", False),
    ("allow_repeated_recalls", True), ("allow_repeated_recalls", False),
    ("stop_probability_scale", .1),
])
def test_optimized_event_and_prepared_loss_reject_unsupported_parameter_names(name, value):
    parameters = OPTIMIZED_CMR_PARAMS | {name: value}
    # Name binding must work with traced values, including a false Boolean.
    with pytest.raises(TypeError, match=name):
        jax.jit(lambda p: CMR(3, p).context.state)(parameters)
    data = make_dataset([1, 2, 0], pres_itemnos=[1, 2, 3])
    prepared = OptimizedFactory(data).prepare_predictions(jnp.array([0]), data["recalls"] != 0)
    # A purported free parameter must not disappear from the bulk calculation.
    with pytest.raises(TypeError, match=name):
        jax.jit(lambda x: _sequence_likelihood(prepared, OPTIMIZED_CMR_PARAMS, (name,), x))(
            jnp.array([[value]])
        )
