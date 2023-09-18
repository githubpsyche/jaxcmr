# %% Imports
from functools import partial
import jax
from evosax import DE
from jax.tree_util import Partial
from jax import numpy as jnp, jit
from beartype.typing import Callable
from jaxcmr.helpers import PRNGKeyArray, Bool, Array, get_item_count
import pytest
import json
from jaxcmr.datasets import load_data, generate_trial_mask
from jaxcmr.memorysearch import BaseCMR, _create_predict_fn
from evosax import ParameterReshaper


# %% Dev


def scale_params(params, bounds):
    return jax.tree_util.tree_map(
        lambda x, y: y[0] + jax.nn.sigmoid(x) * (y[1] - y[0]), params, bounds
    )


@partial(jit, static_argnums=(0, 1, 2, 3, 4))
def es_step(
    strategy,
    es_params,
    loss_fn,
    x_to_dict,
    item_count,
    parameters,
    trials,
    state_input,
    tmp,
):
    """Helper es step to lax.scan through."""
    rng, state = state_input
    rng, rng_iter = jax.random.split(rng)
    x, state = strategy.ask(rng_iter, state, es_params)
    x_dict = scale_params(x_to_dict.reshape(x), parameters["free"])
    fitness = loss_fn(model_create_fn, item_count, trials, x_dict)
    state = strategy.tell(x, fitness, state, es_params)
    return [rng, state], fitness[jnp.argmin(fitness)]


def fit_to_h5(
    loss_fn: Callable,
    model_create_fn: Callable,
    data: dict,
    parameters: list[dict],
    rng: PRNGKeyArray,
    trial_mask: Bool[Array, "trial_count"],
    num_steps: int,
):
    parameters = parameters[0]
    x_to_dict = ParameterReshaper(
        {key: parameters["fixed"][key] for key in parameters["free"]}
    )
    strategy = DE(popsize=15, num_dims=len(parameters["free"]))
    es_params = strategy.default_params
    trials = data["recalls"][trial_mask]
    presentations = data["pres_itemnos"][trial_mask]
    item_count = get_item_count(presentations[0])

    _es_step = Partial(
        es_step, strategy, es_params, loss_fn, x_to_dict, item_count, parameters, trials
    )

    with jax.disable_jit():
        _, scan_out = jax.lax.scan(
            _es_step,
            [rng, strategy.initialize(rng, es_params)],
            [jnp.zeros(num_steps)],
        )
    return jnp.min(scan_out)


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


def test_peers_model_fit(
    model_create_fn, peers_data, peers_parameters, rng, peers_trial_query
):
    trial_mask = generate_trial_mask(peers_data, peers_trial_query)
    fit_result = fit_to_h5(
        _create_predict_fn,
        model_create_fn,
        peers_data,
        peers_parameters,
        rng,
        trial_mask,
    )
