"""
Tests for Dataset Simulation

This module contains tests for the trial simulation functions in
jaxcmr/memorysearch/simulate.py.

The ultimate objective of this functionality is the ability to rapidly simulate entire datasets
from a given model and multiple parameter sets in a single jax operation.

The big obstacle to this is the potential heterogeneity of item counts across trials.
Since these shape key model structures, jax makes it difficult to vectorize across trials.

"""

# %% Imports

from jaxcmr.memorysearch import BaseCMR, InstanceCMR, simulate_h5_from_h5
import pytest
import jax
from jaxcmr.datasets import load_data, generate_trial_mask
import json
import matplotlib.pyplot as plt
from jaxcmr.analyses import plot_crp, plot_pfr, plot_spc

# %% Fixtures

data_path = "D:/data/{}.h5"
param_path = "D:/data/results/{}_{}_{}.jsonl"


@pytest.fixture
def peers_data_tag():
    return "HealyKahana2014"


@pytest.fixture
def peers_trial_query():
    return "data['listtype'] == -1"


@pytest.fixture
def lohnas_data_tag():
    return "LohnasKahana2014"


@pytest.fixture
def lohnas_trial_query():
    return "data['list_type'] != -1"


@pytest.fixture
def model_name():
    return "Base_CMR"


@pytest.fixture
def model_create_fn(model_name):
    return BaseCMR


@pytest.fixture
def peers_data(peers_data_tag):
    return load_data(data_path.format(peers_data_tag))


@pytest.fixture
def lohnas_data(lohnas_data_tag):
    return load_data(data_path.format(lohnas_data_tag))


@pytest.fixture
def peers_parameters(model_name, peers_data_tag):
    ignore_first_recall = False
    with open(param_path.format(model_name, peers_data_tag, ignore_first_recall)) as f:
        param_list = [json.loads(line) for line in f.readlines()]
    return param_list


@pytest.fixture
def lohnas_parameters(model_name, lohnas_data_tag):
    ignore_first_recall = False
    with open(param_path.format(model_name, lohnas_data_tag, ignore_first_recall)) as f:
        param_list = [json.loads(line) for line in f.readlines()]
    return param_list


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# %% Tests

@pytest.mark.parametrize("model_create_fn", [BaseCMR, InstanceCMR])
def test_data_simulation(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query
):
    trial_mask = generate_trial_mask(peers_data, peers_trial_query)
    sim_h5 = simulate_h5_from_h5(
        model_create_fn, peers_data, peers_parameters, rng, trial_mask, 1
    )
    assert sim_h5["recalls"].shape == peers_data["recalls"][trial_mask].shape


def test_peers_data_simulation(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query
):
    trial_mask = generate_trial_mask(peers_data, peers_trial_query)
    sim_h5 = simulate_h5_from_h5(
        model_create_fn, peers_data, peers_parameters, rng, trial_mask, 1
    )
    assert sim_h5["recalls"].shape == peers_data["recalls"][trial_mask].shape


def test_peers_multi_experiment_simulation(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query
):
    experiment_count = 10
    trial_mask = generate_trial_mask(peers_data, peers_trial_query)
    sim_h5 = simulate_h5_from_h5(
        model_create_fn, peers_data, peers_parameters, rng, trial_mask, experiment_count
    )
    assert sim_h5["recalls"].shape[0] == (
        peers_data["recalls"][trial_mask].shape[0] * experiment_count
        )


def test_peers_spc(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query, model_name
):
    data, parameters, trial_query = peers_data, peers_parameters, peers_trial_query
    trial_mask = generate_trial_mask(data, trial_query)
    result = simulate_h5_from_h5(model_create_fn, data, parameters, rng, trial_mask, 10)
    datasets = [data, result]
    trial_masks = [trial_mask, generate_trial_mask(result, trial_query)]

    axis = plot_spc(
        datasets,
        trial_masks,
        labels=["Data", "Model"],
        contrast_name=model_name.replace("_", " "),
    )

    axis.legend(title=None)
    axis.tick_params(labelsize=14)
    axis.set_xlabel(axis.get_xlabel(), fontsize=16)
    axis.set_ylabel(axis.get_ylabel(), fontsize=16)

    plt.show()


def test_lohnas_spc(
    model_create_fn, lohnas_data, lohnas_parameters, rng, lohnas_trial_query, model_name
):
    data, parameters, trial_query = lohnas_data, lohnas_parameters, lohnas_trial_query
    trial_mask = generate_trial_mask(data, trial_query)
    result = simulate_h5_from_h5(model_create_fn, data, parameters, rng, trial_mask, 10)
    datasets = [data, result]
    trial_masks = [trial_mask, generate_trial_mask(result, trial_query)]

    axis = plot_spc(
        datasets,
        trial_masks,
        labels=["Data", "Model"],
        contrast_name=model_name.replace("_", " "),
    )

    axis.legend(title=None)
    axis.tick_params(labelsize=14)
    axis.set_xlabel(axis.get_xlabel(), fontsize=16)
    axis.set_ylabel(axis.get_ylabel(), fontsize=16)

    plt.show()


def test_pfr(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query, model_name
):
    data, parameters, trial_query = peers_data, peers_parameters, peers_trial_query
    trial_mask = generate_trial_mask(data, trial_query)
    result = simulate_h5_from_h5(model_create_fn, data, parameters, rng, trial_mask, 1)
    datasets = [data, result]
    trial_masks = [trial_mask, generate_trial_mask(result, trial_query)]

    axis = plot_pfr(
        datasets,
        trial_masks,
        labels=["Data", "Model"],
        contrast_name=model_name.replace("_", " "),
    )

    axis.legend(title=None)
    axis.tick_params(labelsize=14)
    axis.set_xlabel(axis.get_xlabel(), fontsize=16)
    axis.set_ylabel(axis.get_ylabel(), fontsize=16)

    plt.show()


def test_crp(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query, model_name
):
    data, parameters, trial_query = peers_data, peers_parameters, peers_trial_query
    trial_mask = generate_trial_mask(data, trial_query)
    result = simulate_h5_from_h5(model_create_fn, data, parameters, rng, trial_mask, 1)
    datasets = [data, result]
    trial_masks = [trial_mask, generate_trial_mask(result, trial_query)]

    axis = plot_crp(
        datasets,
        trial_masks,
        labels=["Data", "Model"],
        contrast_name=model_name.replace("_", " "),
    )

    axis.legend(title=None)
    axis.tick_params(labelsize=14)
    axis.set_xlabel(axis.get_xlabel(), fontsize=16)
    axis.set_ylabel(axis.get_ylabel(), fontsize=16)

    plt.show()
