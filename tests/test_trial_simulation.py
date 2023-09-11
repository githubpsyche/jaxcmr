"""
Tests for Trial Simulation

This module contains tests for the trial simulation functions in
jaxcmr/memorysearch/simulate.py.

The ultimate objective of this functionality is the ability to rapidly simulate entire datasets
from a given model and multiple parameter sets in a single jax operation.

The big obstacle to this is the potential heterogeneity of item counts across trials.
Since these shape key model structures, jax makes it difficult to vectorize across trials.

"""

# %% Imports

from jaxcmr.memorysearch import (
    BaseCMR,
    simulate_trial,
    simulate_trials,
)
from jaxcmr.helpers import tree_transpose
import pytest
import jax
from jax import numpy as jnp

# %% Fixtures


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
    }


@pytest.fixture
def parameter_list(parameters):
    return [parameters, parameters]


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def presentations():
    """Two presentation sequences with different study order but same item count"""
    return jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])


# %% Tests


def test_single_trial_simulation(parameters, rng):
    trial = simulate_trial(BaseCMR, 10, rng, parameters)
    assert trial.shape[0] == 10


def test_multiple_trial_simulation(parameters, rng):
    trials = jax.vmap(simulate_trial, in_axes=(None, None, 0, None))(
        BaseCMR, 10, jax.random.split(rng, 100), parameters
    )
    assert trials.shape[0] == 100
    assert trials.shape[1] == 10


def test_multiple_presentation_trial_simulation(parameters, rng, presentations):
    trials = jax.vmap(simulate_trial, in_axes=(None, None, 0, None, None))(
        BaseCMR, 10, presentations, rng, parameters
    )
    assert trials.shape[0] == 2
    assert trials.shape[1] == 10
    assert not jnp.allclose(trials[0], trials[1])

    # with jax.disable_jit():
    trial = simulate_trial(BaseCMR, 10, presentations[0], rng, parameters)

    nonzero_elements = trial[trial != 0]
    unique_nonzero_elements = jnp.unique(nonzero_elements)
    assert len(nonzero_elements) == len(unique_nonzero_elements)


def test_tree_transpose(parameter_list):
    result = tree_transpose(parameter_list)
    assert len(result["encoding_drift_rate"]) == 2


def test_multiple_parameter_trial_simulation(parameter_list, rng):
    trials = jax.vmap(simulate_trial, in_axes=(None, None, 0, 0))(
        BaseCMR,
        10,
        jax.random.split(rng, len(parameter_list)),
        tree_transpose(parameter_list),
    )
    assert trials.shape[0] == len(parameter_list)
    assert trials.shape[1] == 10


def test_multi_parameter_multi_trial_simulation(parameter_list, rng):
    _simulate_trials = jax.vmap(simulate_trials, in_axes=(None, None, None, 0, 0))
    trials = _simulate_trials(
        BaseCMR,
        10,
        100,
        jax.random.split(rng, len(parameter_list)),
        tree_transpose(parameter_list),
    )
    assert trials.shape[0] == len(parameter_list)
    assert trials.shape[1] == 100
    assert trials.shape[2] == 10

    assert not jnp.allclose(trials[0, 0], trials[1, 0])
