from jaxcmr.memorysearch import (
    BaseCMR,
    InstanceCMR,
    uniform_presentations_data_likelihood,
    variable_presentations_data_likelihood,
    predict_and_simulate_trial,
    predict_and_simulate_pres_and_trial,
    log_likelihood,
    start_retrieving,
    experience,
    trial_item_count,
    trial_list_length,
)
from jax import numpy as jnp, jit, lax, vmap, disable_jit


class TestDataLikelihood:
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
        "mcf_trace_sensitivity": 1.0,
    }

    item_count = 10
    max_presentation_count = 10
    presentation = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]]
    )
    trial = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def test_uniform_data_likelihood(self):
        model = BaseCMR.create(self.item_count, self.parameters)
        model = start_retrieving(experience(model))
        likelihood = log_likelihood(
            vmap(predict_and_simulate_trial, in_axes=(None, 0))(model, self.trial)[1]
        )

        uniform_likelihood = uniform_presentations_data_likelihood(
            BaseCMR.create, self.item_count, self.trial, self.parameters
        )

        assert jnp.allclose(likelihood, uniform_likelihood)

    def test_predict_and_simulate_pres(self):

        item_counts = vmap(trial_item_count)(self.presentation)

        likelihood = predict_and_simulate_pres_and_trial(
            InstanceCMR.create,
            item_counts[0].item(),
            self.presentation[0],
            self.trial[0],
            self.parameters,
        )
        likelihood

    def test_variable_data_likelihood(self):

        item_counts = vmap(trial_item_count)(self.presentation)

        likelihood = []
        for trial_index, item_count in enumerate(item_counts):
            likelihood.append(variable_presentations_data_likelihood(
                InstanceCMR.create,
                item_count.item(),
                self.presentation[trial_index],
                self.trial[trial_index],
                self.parameters,
            ))
        likelihood

    def test_multitrial_variable_data_likelihood(self):

        item_counts = vmap(trial_item_count)(self.presentation)

        likelihood = []
        for item_count in jnp.unique(item_counts):
            likelihood.append(variable_presentations_data_likelihood(
                InstanceCMR.create,
                item_count.item(),
                self.presentation[item_counts==item_count],
                self.trial[item_counts==item_count],
                self.parameters,
            ))
        assert jnp.allclose(sum(likelihood), 22.788351)

    def test_dispatch_variable_data_likelihood(self):

        item_counts = vmap(trial_item_count)(self.presentation)

        likelihood_fn = variable_presentations_data_likelihood(
            InstanceCMR.create,
            item_counts,
            self.presentation,
            self.trial,
        )
        likelihood = likelihood_fn(self.parameters)
        assert jnp.allclose(likelihood, 22.788351)