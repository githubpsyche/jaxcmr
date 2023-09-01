from jaxcmr.memorysearch import (
    InstanceCMR,
    BaseCMR,
    experience,
    retrieve,
    start_retrieving,
    item_probability,
    outcome_probability,
    stop_probability,
    predict_and_simulate_trial,
    uniform_presentations_data_likelihood
)
from jax import numpy as jnp, jit, lax, disable_jit
import pytest

@pytest.fixture
def parameters():
    return {
        "encoding_drift_rate": 0.8016327576683261,
        "delay_drift_rate": 0.9966411723460118,
        "start_drift_rate": 0.051123130268380085,
        "recall_drift_rate": 0.8666706252504806,
        "shared_support": 0.016122091797498662,
        "item_support": 0.8877852952105489,
        "learning_rate": 0.10455606050373444,
        "primacy_scale": 33.57091895097917,
        "primacy_decay": 1.57091895097917,
        "stop_probability_scale": 0.0034489993376706257,
        "stop_probability_growth": 0.3779780110633191,
        "choice_sensitivity": 1.0,
        "mcf_trace_sensitivity": 1.0,
        "mfc_trace_sensitivity": 1.0
    }

@pytest.fixture
def base_cmr(parameters):
    return BaseCMR(10, 10, parameters)

@pytest.fixture
def cmr(parameters) -> InstanceCMR:
    return InstanceCMR(10, 10, parameters)

def test_init_instance_cmr(cmr):
    cmr

def test_has_choice_sensitivity(cmr, parameters):
    assert cmr.mcf.feature_scale == parameters["choice_sensitivity"]

def test_stop_probability(cmr):
    cmr = experience(cmr)
    cmr = retrieve(cmr, 1)
    p_stop = stop_probability(cmr)
    assert jnp.allclose(p_stop, 0.005033231)
    assert jnp.allclose(outcome_probability(cmr, 0), 0.005033231)

def test_outcome_probabilities(cmr, base_cmr):
    cmr = experience(cmr, 2)
    cmr = retrieve(cmr, 1)
    instance_p_all = outcome_probability(cmr)

    cmr = experience(base_cmr, 2)
    cmr = retrieve(cmr, 1)
    base_p_all = outcome_probability(cmr)

    assert jnp.allclose(instance_p_all, base_p_all)
