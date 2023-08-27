from beartype.typing import Callable
from jaxcmr.helpers import Integer, Float, Array, ScalarInteger, PRNGKeyArray, study_events, recall_events
from jaxcmr.memorysearch import experience, start_retrieving, free_recall, BaseCMR, trial_item_count, simulate_trial
from jaxcmr.analyses import multi_pres_spc
import jax
from jax import numpy as jnp
import numpy as np

import pytest
from jaxcmr.datasets import load_data
import json
from numba import njit


# %% Dev Code

@njit(nogil=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1



# %% Tests

class TestDataSimulation:
    model_name = 'BaseCMR'
    data_tag = 'HealyKahana2014' #'LohnasKahana2014'
    ignore_first_recall = False

    data_path = '../../../../data/{}.h5'
    data_path = data_path.format(data_tag)
    data = load_data(data_path)

    presentations = data['pres_itemnos']
    trials = data['recalls']
    list_types = data['listtype'].flatten() #data['list_type'].flatten()
    subjects = data['subject'].flatten()

    result_path = '../../../../data/results/jax_{}_{}_{}.jsonl'
    result_path = result_path.format(model_name, data_tag, ignore_first_recall)
    result_path = r"D:\data\results\jax_Base_CMR_HealyKahana2014_False.jsonl"

    with open(result_path) as f:
        parameter_sets = [json.loads(line) for line in f.readlines()]

    def test_simulate_trial(self):
        model_create_fn = BaseCMR.create
        presentation = jnp.array(self.presentations[0])
        item_count = trial_item_count(presentation).item()
        rng = jax.random.PRNGKey(0)
        parameters = self.parameter_sets[0]['fixed']

        recalls = simulate_trial(
            model_create_fn, item_count,
            presentation, rng, parameters)

        assert recalls.shape[0] == item_count
        assert jnp.all(recalls <= item_count)
        assert jnp.all(recalls >= 0)

        recalls2 = simulate_trial(
            model_create_fn, item_count,
            presentation, rng, parameters)

        assert jnp.allclose(recalls, recalls2)

        recalls3 = simulate_trial(
            model_create_fn, item_count,
            presentation, jax.random.split(rng)[0], parameters)

        assert not jnp.allclose(recalls, recalls3)

    def test_simulate_trials(self):
        experiment_count = 100
        simulate_trials = jax.vmap(simulate_trial, in_axes=(None, None, 0, 0, None))
        mask = np.logical_and(self.list_types == -1, self.subjects == self.subjects[0])
        presentations = self.presentations[mask]
        trials = self.trials[mask]
        item_count = trial_item_count(presentations[0]).item()
        rng = jax.random.split(jax.random.PRNGKey(0), presentations.shape[0] * experiment_count)
        parameters = self.parameter_sets[0]['fixed']

        recalls = simulate_trials(
            BaseCMR.create, item_count,
            jnp.repeat(presentations, experiment_count, 0),
            rng, parameters
        )

        assert recalls.shape[0] == presentations.shape[0] * experiment_count
        assert recalls.shape[1] == item_count
        assert jnp.all(recalls <= item_count)
        assert jnp.all(recalls >= 0)
        assert not jnp.allclose(recalls[0], recalls[1])

        data_spc = multi_pres_spc(np.array(trials), np.array(presentations),
                                  self.presentations.shape[1])
        sim_spc = multi_pres_spc(
            np.array(recalls),
            np.array(jnp.repeat(presentations, experiment_count, 0)),
            self.presentations.shape[1])
        assert jnp.allclose(data_spc, sim_spc, atol=0.05)
