# %% Imports

from jaxcmr.memorysearch import CMR, BaseCMR, _retrieve_item, predict_trial, experience, retrieve
from jaxcmr.memory import InstanceMcf, InstanceMfc, probe
from jaxcmr.context import TemporalContext, integrate
from jaxcmr.helpers import ScalarInteger, replace
from jax import numpy as jnp

import pytest

# %% Setup

class TestModel(CMR, mutable=True):
    pass

def TestCMR(
    item_count,
    presentation_count,
    parameters: dict,
) -> TestModel:
    
    return TestModel.create(
        InstanceMfc.create,
        InstanceMcf.create,
        TemporalContext.create,
        item_count,
        presentation_count,
        parameters,
    )

@_retrieve_item.dispatch
def _retrieve_item(model: TestModel, choice: ScalarInteger) -> CMR:
    """Retrieve item with index choice-1"""
    context_input = probe(model.mfc, model.items[choice - 1])
    return replace(
        model,
        context=integrate(model.context, context_input, model.recall_drift_rate),
        recall_sequence=model.recall_sequence.at[model.recall_total].set(choice - 1),
        recall_mask=model.recall_mask.at[choice - 1].set(False),
        recall_total=model.recall_total + 1,
    )

# %% Fixtures

parameters = {
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
    "mfc_choice_sensitivity": 1.0,
    "mfc_trace_sensitivity": 1.0,
    "mcf_trace_sensitivity": 1.0,
}

item_count = 16
presentation_count = 16

@pytest.fixture
def cmr_variant():
    return TestCMR(
        item_count,
        presentation_count,
        parameters,
    )

@pytest.fixture
def base_cmr():
     return BaseCMR(item_count, presentation_count, parameters)

# %% Tests

def test_variant_instantiation(cmr_variant, base_cmr):
    assert isinstance(cmr_variant, TestModel)
    assert isinstance(cmr_variant, CMR)
    assert isinstance(base_cmr, CMR)

def test_variant_experience(cmr_variant, base_cmr):
    base_cmr = experience(base_cmr)
    cmr_variant = experience(cmr_variant)
    
    assert jnp.allclose(cmr_variant.context.state, base_cmr.context.state)

def test_variant_retrieve(cmr_variant, base_cmr):
    base_cmr = experience(base_cmr)
    cmr_variant = experience(cmr_variant)
    
    base_cmr = retrieve(base_cmr, 1)
    cmr_variant = retrieve(cmr_variant, 1)
    
    assert jnp.allclose(cmr_variant.context.state, base_cmr.context.state)
    assert jnp.allclose(cmr_variant.recall_sequence, base_cmr.recall_sequence)
    assert jnp.allclose(cmr_variant.recall_mask, base_cmr.recall_mask)
    assert jnp.allclose(cmr_variant.recall_total, base_cmr.recall_total)