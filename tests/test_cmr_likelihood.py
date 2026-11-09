"""Scientific parity and selection safety for the optimized regular-CMR loss."""

from support import CMR_PARAMS
import jax
import jax.numpy as jnp
import pytest
from jaxcmr.components import context, linear_memory
from jaxcmr.components.termination import NoStopTermination
from jaxcmr.helpers import make_dataset
from jaxcmr.loss.cmr_likelihood import FastCMRLikelihoodLoss
from jaxcmr.models.cmr import make_factory
from jaxcmr.models.optimized_cmr import ModelFactory as OptimizedFactory

FACTORY = make_factory(
    linear_memory.init_mfc, linear_memory.init_mcf, context.init, NoStopTermination
)
PARAMS = dict(CMR_PARAMS)
NAMES = tuple((name for name, value in PARAMS.items() if not isinstance(value, bool)))


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
        ("recalls", jnp.zeros((1, 40), dtype=float)),
    ],
)
def test_unsupported_observations_fail_explicitly(field, value):
    with pytest.raises(ValueError):
        FastCMRLikelihoodLoss(OptimizedFactory, {**repeated_trial(), field: value}, None)


@pytest.mark.parametrize(
    "issue",
    ["features", "learn_after_context_update", "allow_repeated_recalls", "parameter mismatch"],
)
def test_optimized_model_rejects_unsupported_features_and_parameters(issue):
    data = repeated_trial()
    if issue == "features":
        with pytest.raises(ValueError, match="features must be None"):
            FastCMRLikelihoodLoss(OptimizedFactory, data, jnp.eye(40))
        return
    prepared = FastCMRLikelihoodLoss(OptimizedFactory, data, None).prepare_trials(jnp.array([0]))
    point = jnp.array([[2.0]])
    with pytest.raises(ValueError, match=issue):
        if issue == "parameter mismatch":
            prepared(PARAMS, ("stop_probability_scale",), point)
        else:
            jax.jit(lambda p: prepared(PARAMS | {issue: True}, ("choice_sensitivity",), p))(point)
