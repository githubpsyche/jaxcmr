# %% Imports
from jaxcmr.memorysearch import FittingStrategy, BaseCMR, InstanceCMR, create_predict_fn
import pytest
import jax
import json
from jax import numpy as jnp, vmap
from jaxcmr.datasets import load_data, generate_trial_mask

# %% Fixtures

parameter_path = "D:/data/base_cmr_parameters.json"
data_path = "D:/data/{}.h5"

@pytest.fixture
def model_create_fn():
    return BaseCMR

@pytest.fixture
def data_tag():
    return "HealyKahana2014"


@pytest.fixture
def trial_query():
    return "data['listtype'] == -1"


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def parameters():
    with open(parameter_path, "r") as f:
        parameters = json.load(f)
    return parameters


@pytest.fixture
def data(data_tag):
    return load_data(data_path.format(data_tag))

@pytest.fixture
def trial_mask(data, trial_query):
    return generate_trial_mask(data, trial_query)


@pytest.fixture
def subject_trials(data, trial_mask):
    return jnp.array(data["recalls"][trial_mask].reshape(126, 28, 16))


@pytest.fixture
def subject_presentations(data, trial_mask):
    return jnp.array(data["pres_itemnos"][trial_mask].reshape(126, 28, 16))


@pytest.fixture
def single_subject_objective_fn(model_create_fn, subject_presentations, subject_trials):
    return vmap(create_predict_fn(model_create_fn, subject_presentations[0], subject_trials[0]))


# %% Tests


def test_fitting_strategy(single_subject_objective_fn, parameters, rng):
        
    # with jax.disable_jit():
    fitting_fn = FittingStrategy(
        single_subject_objective_fn,
        parameters,
        maxiter=100,
        popsize=15,
        mutation=1.0,
        recombination=0.7,
        disp=True,
    )

    fit_result = fitting_fn(rng)
    fitted_parameters = fitting_fn.format_fit_result(fit_result)
    assert fitted_parameters['likelihood'] < 700
