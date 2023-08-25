from jaxcmr.memorysearch import (
    BaseCMR,
    experience,
    experience_item,
    retrieve,
    start_retrieving,
    item_probability,
    outcome_probability,
    stop_probability,
    outcome_probabilities,
    predict_and_simulate_trial,
    uniform_presentations_data_likelihood
)
from jax import numpy as jnp, jit, lax


class TestBaseCMR:
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

    cmr = jit(lambda parameters: BaseCMR.create(10, 10, parameters))(parameters)

    def test_init_base_cmr(self):
        self.cmr

    def test_has_choice_sensitivity(self):
        assert self.cmr.mcf.choice_sensitivity == self.parameters["choice_sensitivity"]

    def test_experience_item(self):
        cmr = experience_item(self.cmr, 0)
        cmr

    def test_experience_cmr(self):
        cmr = experience(self.cmr)
        cmr

    def test_recall_mask(self):
        cmr = experience(self.cmr)
        cmr = retrieve(cmr, 1)
        retrieved_item_recall_probability = item_probability(cmr, 1)
        assert retrieved_item_recall_probability == 0.0
        assert outcome_probability(cmr, 1) == 0.0

    def test_stop_probability(self):
        cmr = experience(self.cmr)
        cmr = retrieve(cmr, 1)
        p_stop = stop_probability(cmr)
        assert jnp.allclose(p_stop, 0.005033231)
        assert jnp.allclose(outcome_probability(cmr, 0), 0.005033231)

    def test_outcome_probabilities(self):
        cmr = experience(self.cmr, 1)
        cmr = experience(cmr, 2)
        cmr = retrieve(cmr, 1)

        p_all = outcome_probabilities(cmr)
        desired_result = jnp.array(
            [
                0.00503323,
                0.0,
                0.50872654,
                0.101161,
                0.0395892,
                0.02791035,
                0.02784999,
                0.03489236,
                0.0503365,
                0.07826934,
                0.12623152,
            ]
        )
        assert jnp.allclose(p_all.sum(), 1)
        assert jnp.allclose(p_all, desired_result)


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

        model = BaseCMR.create(item_count, item_count, self.parameters['fixed'])
        model = experience(model)

        recalls = np.array(
            lax.map(lambda rng: free_recall(model, rng), jax.random.split(rng, experiment_count))[1]
        )

        spc = single_pres_spc(recalls, item_count)
        pfr = single_pres_pfr(recalls, item_count)
        crp = single_pres_crp(recalls, item_count)

        raise NotImplementedError

    def test_predict_and_simulate_trial(self):
        item_count = 16
        rng = jax.random.PRNGKey(0)
        model = experience(BaseCMR.create(item_count, item_count, self.parameters['fixed']))
        model = start_retrieving(model)
        trial_likelihood = predict_and_simulate_trial(model, self.trials[0][1])
        multitrial_likelihood = lax.map(lambda trial: predict_and_simulate_trial(model, trial)[1], self.trials[0])

        scalar_likelihood = -jnp.sum(jnp.log(multitrial_likelihood))
        assert jnp.allclose(scalar_likelihood, 88686.25)

    def test_uniform_data_likelihood(self):
        item_count = 16
        rng = jax.random.PRNGKey(0)
        model = BaseCMR.create(item_count, item_count, self.parameters['fixed'])
        likelihood = uniform_presentations_data_likelihood(model, self.trials[0])
        assert jnp.allclose(likelihood, 88686.25)
