# %% Imports
import pytest
from jaxcmr.memorysearch import (
    BaseCMR,
    InstanceCMR,
    start_retrieving,
    experience,
    predict_and_simulate_retrieval,
    predict_trial,
    init_and_predict_trial,
    predict_trials,
    _create_predict_fn,
)
from jaxcmr.helpers import log_likelihood, get_item_count
from jax import vmap, numpy as jnp

# %% Fixtures

pytestmark = pytest.mark.parametrize("model_create_fn", [BaseCMR, InstanceCMR])


@pytest.fixture
def uniform_presentations():
    return jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
    )


@pytest.fixture
def variable_presentations():
    return jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
        ]
    )


@pytest.fixture
def trials():
    return jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


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
    }


@pytest.fixture
def item_count():
    return 10


# %% Tests


def test_termination_probability(model_create_fn, item_count, parameters):
    """Test that predict_and_simulate_retrieval can simulate termination"""
    model = start_retrieving(
        experience(model_create_fn(item_count, item_count, parameters))
    )
    updated_model, outcome_probability = predict_and_simulate_retrieval(model, 0)
    assert predict_and_simulate_retrieval(updated_model, 0)[1] == 1.0
    assert predict_and_simulate_retrieval(updated_model, 1)[1] == 0.0


def test_predict_trial(model_create_fn, item_count, parameters, trials):
    """Test that predict_trial tracks termination state and probabilities across a trial"""
    model = start_retrieving(
        experience(model_create_fn(item_count, item_count, parameters))
    )

    _, trial_probabilities = predict_trial(model, trials[0])
    assert trial_probabilities.shape == trials[0].shape
    assert (trial_probabilities == 1.0).sum() + 1 == (trials[0] == 0).sum()

def test_predict_trial_with_higher_item_count(
        model_create_fn, item_count, parameters, trials, uniform_presentations):
    """
    Test that init_and_predict_trial makes the same predictions with higher than needed item count
    """

    model = start_retrieving(
        experience(model_create_fn(item_count, item_count, parameters))
    )
    _, base_trial_probabilities = predict_trial(model, trials[0])

    _, init_trial_probabilities = init_and_predict_trial(
        model_create_fn, item_count, uniform_presentations[0], trials[0], parameters
    )

    _, higher_init_trial_probabilities = init_and_predict_trial(
        model_create_fn, item_count + 10, uniform_presentations[0], trials[0], parameters
    )

    assert jnp.allclose(base_trial_probabilities, init_trial_probabilities)
    assert jnp.allclose(base_trial_probabilities, higher_init_trial_probabilities)



def test_predict_trials(model_create_fn, item_count, parameters, trials):
    """Test that sum of predict_trial calls is same result as a single predict_trials call"""
    model = start_retrieving(
        experience(model_create_fn(item_count, item_count, parameters))
    )

    multicall_result = sum(
        log_likelihood(predict_trial(model, trial)[1]) for trial in trials
    )
    assert multicall_result == predict_trials(
        model_create_fn, item_count, trials, parameters
    )


def test_factory_fn(
    model_create_fn, item_count, parameters, uniform_presentations, trials
):
    """Test that factory function can create a predict function that works like predict_trials"""
    predict_fn = _create_predict_fn(model_create_fn, uniform_presentations, trials)
    predict_result = predict_fn(parameters)
    assert predict_result == predict_trials(
        model_create_fn, item_count, trials, parameters
    )


def test_dispatch_predict_trials(
    model_create_fn, item_count, parameters, uniform_presentations, trials
):
    """Test that predict_trials returns same result when a uniform presentations array is passed"""
    predict_result = predict_trials(
        model_create_fn, item_count, uniform_presentations, trials, parameters
    )
    assert predict_result == predict_trials(
        model_create_fn, item_count, trials, parameters
    )


def test_variable_presentations_predict(
    model_create_fn, parameters, variable_presentations, trials
):
    """Test that presentations with variable item_counts can be handled w/o dispatch"""

    item_counts = vmap(get_item_count)(variable_presentations)
    multicall_result = sum(
        log_likelihood(
            init_and_predict_trial(
                model_create_fn,
                item_counts[i].item(),
                variable_presentations[i],
                trials[i],
                parameters,
            )[1]
        )
        for i in range(len(item_counts))
    )

    max_item_count = max(item_counts).item()
    predict_result = predict_trials(
        model_create_fn, max_item_count, variable_presentations, trials, parameters
    )

    assert jnp.allclose(multicall_result, predict_result)


def test_dispatch_variable_presentations_predict(
    model_create_fn, parameters, variable_presentations, trials
):
    """Test that create_predict_fn produces fn that handles variable presentations correctly"""

    item_counts = vmap(get_item_count)(variable_presentations)
    multicall_result = sum(
        log_likelihood(
            init_and_predict_trial(
                model_create_fn,
                item_counts[i].item(),
                variable_presentations[i],
                trials[i],
                parameters,
            )[1]
        )
        for i in range(len(item_counts))
    )

    predict_fn = _create_predict_fn(model_create_fn, variable_presentations, trials)
    predict_result = predict_fn(parameters)

    assert jnp.allclose(multicall_result, predict_result)

