from jaxcmr.memorysearch import (
    BaseCMR,
    experience,
    retrieve,
    start_retrieving,
    item_probability,
    stop_probability,
    outcome_probability,
    predict_trials
)
from jax import numpy as jnp, lax
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
    }


@pytest.fixture
def cmr(parameters):
    """Base CMR instance with 10 items presented once each"""
    return BaseCMR(10, 10, parameters)


def test_init_base_cmr(cmr):
    """Check that the base CMR instance is produced without error"""
    pass


def test_has_choice_sensitivity(cmr, parameters):
    """Check that the base CMR instance passes choice sensitivity to mcf attribute"""
    assert cmr.mcf.choice_sensitivity == parameters["choice_sensitivity"]


def test_experience_item(cmr):
    """Check that the base CMR instance experiences an item without error"""
    experience(cmr, 1)


def test_experience_cmr(cmr):
    """Check that the base CMR instance experiences its item set without error"""
    experience(cmr)


def test_recall_mask(cmr):
    """Check that Base CMR ascribes 0 probability to retrieval of already recalled items"""
    cmr = experience(cmr)
    cmr = retrieve(cmr, 1)
    retrieved_item_recall_probability = item_probability(cmr, 1)
    assert retrieved_item_recall_probability == 0.0
    assert outcome_probability(cmr, 1) == 0.0


def test_stop_probability(cmr):
    """Check that Base CMR ascribes correct termination probability after retrieving one item"""
    cmr = experience(cmr)
    cmr = retrieve(cmr, 1)
    p_stop = stop_probability(cmr)
    assert jnp.allclose(p_stop, 0.005033231)
    assert jnp.allclose(outcome_probability(cmr, 0), 0.005033231)


def test_outcome_probabilities(cmr):
    raise NotImplementedError


# %%

from jaxcmr.memorysearch import free_recall
from jaxcmr.evaluation import extract_objective_data
from jaxcmr.datasets import load_data, generate_trial_mask, load_parameters
from jaxcmr.analyses import single_pres_spc, single_pres_pfr, single_pres_crp
import numpy as np
import jax


class TestWithData:
    data_path = '../../../data/{}.h5'
    data = load_data(data_path.format('HealyKahana2014'))
    parameters = load_parameters('../../../data/base_cmr_parameters.json')
    parameters['fixed']['mcf_trace_sensitivity'] = 1.0

    trial_query = "data['listtype'] == -1"
    trial_mask = generate_trial_mask(data, trial_query)
    trials, list_lengths, presentations, pres_string_ids, has_repetitions = extract_objective_data(data, trial_mask)

    # %%

    def test_summary_statistics(self):
        experiment_count = 10000
        item_count = 16
        rng = jax.random.PRNGKey(0)

        model = BaseCMR(item_count, item_count, self.parameters['fixed'])
        model = start_retrieving(experience(model))

        recalls = np.array(
            lax.map(lambda rng: free_recall(model, rng), jax.random.split(rng, experiment_count))[1]
        )

        spc = single_pres_spc(recalls, item_count)
        pfr = single_pres_pfr(recalls, item_count)
        crp = single_pres_crp(recalls, item_count)

        raise NotImplementedError


    def test_uniform_data_likelihood(self):
        item_count = 16
        model = BaseCMR(item_count, item_count, self.parameters['fixed'])
        likelihood = predict_trials(BaseCMR, item_count, self.trials[0], self.parameters['fixed'])
        assert jnp.allclose(likelihood, 88686.25)
