from jaxcmr.memorysearch import (
    InstanceCMR,
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


class TestInstanceCMR:
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

    base_cmr = jit(lambda parameters: BaseCMR.create(10, parameters))(parameters)
    cmr = jit(
        lambda parameters: InstanceCMR.create(10, parameters))(parameters)

    def test_has_choice_sensitivity(self):
        assert self.cmr.mcf.feature_scale == self.parameters["choice_sensitivity"]

    def test_stop_probability(self):
        cmr = experience(self.cmr)
        cmr = retrieve(cmr, 1)
        p_stop = stop_probability(cmr)
        assert jnp.allclose(p_stop, 0.005033231)
        assert jnp.allclose(outcome_probability(cmr, 0), 0.005033231)

    def test_outcome_probabilities(self):
        cmr = experience(self.cmr, 2)
        cmr = retrieve(cmr, 1)
        instance_p_all = outcome_probabilities(cmr)

        cmr = experience(self.base_cmr, 2)
        cmr = retrieve(cmr, 1)
        base_p_all = outcome_probabilities(cmr)

        assert jnp.allclose(instance_p_all, base_p_all)
