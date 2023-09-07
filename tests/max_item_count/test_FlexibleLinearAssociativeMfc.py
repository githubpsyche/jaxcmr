from jax import numpy as jnp, vmap, disable_jit
from jaxcmr.memory import (
    LinearAssociativeMcf,
    LinearAssociativeMfc,
    associate,
    probe,
)
from jaxcmr.helpers import power_scale
import plum
import pytest

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
}

def test_single_size_create_by_item_count():
    """Mfc can be initialized with an item count and learning rate"""
    item_count = 16
    LinearAssociativeMfc.create(item_count, parameters["learning_rate"])

def test_erroneous_multi_size_create_by_item_count():

    with pytest.raises(TypeError):
        vmap(LinearAssociativeMfc.create)(jnp.array([5, 10, 15]), jnp.array([.3, .5, .7]))

def test_single_size_create_with_max_size():
    """Mfc can be initialized with an item count and learning rate"""
    item_count = 16
    LinearAssociativeMfc.create(item_count, item_count, parameters["learning_rate"])

def test_multi_size_create():
    _create = vmap(LinearAssociativeMfc.create, in_axes=(0, None, None))
    _create(jnp.array([5, 10, 15]), 15, parameters["learning_rate"])