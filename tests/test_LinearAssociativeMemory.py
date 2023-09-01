from jax import numpy as jnp, jit, disable_jit
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

item_count = 16
contexts = jnp.eye(item_count, item_count + 2, 1)
items = jnp.eye(item_count, dtype=jnp.float32)


def test_initialize_with_item_count():
    "Mfc can be initialized with an item count and learning rate"
    LinearAssociativeMfc.create(item_count, parameters["learning_rate"])


def test_initialize_with_items():
    "Mfc can be initialized with an array of item representations"
    LinearAssociativeMfc.create(items, parameters["learning_rate"])


def test_initialize_with_dict():
    "Mfc can be initialized with a dictionary of parameters"
    LinearAssociativeMfc.create(item_count, parameters)


def test_initialize_with_dict_and_items():
    "Mfc can be initialized with a dict and array of item representations"
    LinearAssociativeMfc.create(items, parameters)


def test_initialize_methods_are_equivalent():
    "Initializing with 16 or jnp.eye(16) produces equivalent memory states"
    memory1 = LinearAssociativeMfc.create(item_count, parameters["learning_rate"])
    memory2 = LinearAssociativeMfc.create(items, parameters["learning_rate"])
    memory3 = LinearAssociativeMfc.create(item_count, parameters)
    memory4 = LinearAssociativeMfc.create(items, parameters)

    assert jnp.array_equal(memory1.state, memory2.state)
    assert jnp.array_equal(memory3.state, memory4.state)
    assert jnp.array_equal(memory1.state, memory3.state)


def test_generalized_mfc():
    "Init w overlapping items connects applicable context and item features."
    items = jnp.eye(item_count, item_count + 4)
    items = items.at[1:, 0].set(0.2)
    memory = LinearAssociativeMfc.create(items, 0.1)
    desired_result = jnp.array(
        [
            0.0,
            0.9,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.17999999,
            0.0,
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(memory.state[0], desired_result)


def test_dispatch_associate():
    "Calls to associate should work for instances of subtypes of LinearAssociativeMemory"
    mfc = LinearAssociativeMfc.create(item_count, parameters)
    mcf = LinearAssociativeMcf.create(item_count, parameters)
    mfc = associate(mfc, 0.1, items[0], contexts[0])
    mcf = associate(mcf, 0.1, contexts[0], items[0])


def test_shape_selective_dispatch():
    "Associating with wrong-shaped inputs raises NotFoundLookupError"

    memory = LinearAssociativeMfc.create(item_count, parameters)
    with pytest.raises(plum.NotFoundLookupError):
        associate(memory, items[0], items[0])


def test_probe_result_size():
    "Probing should return a result of item_count + 2 dimensions"
    memory = LinearAssociativeMfc.create(item_count, parameters)
    result = probe(memory, items[0])
    assert result.shape[0] == item_count + 2


def test_choice_sensitivity_scales_activations():
    "Choice sensitivity scales activations"
    mcf = LinearAssociativeMcf.create(
        item_count, parameters["shared_support"], parameters["item_support"], 1.0
    )
    mcf2 = LinearAssociativeMcf.create(
        item_count, parameters["shared_support"], parameters["item_support"], 3.0
    )

    activation1 = probe(mcf, contexts[0])
    activation2 = probe(mcf2, contexts[0])

    assert jnp.allclose(activation2, power_scale(activation1, 3.0))

    assert jnp.allclose(
        activation2 / jnp.sum(activation2),
        jnp.power(activation1, 3.0) / jnp.sum(jnp.power(activation1, 3.0)),
    )
