from jaxcmr.memorysearch import BaseCMR, experience, experience_item, retrieve, item_probability, outcome_probability, stop_probability, outcome_probabilities
from jax import numpy as jnp, jit

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

    cmr = BaseCMR.create(10, 10, parameters)

    def test_init_base_cmr(self):
        self.cmr

    def test_has_choice_sensitivity(self):
        assert(self.cmr.mcf.choice_sensitivity == self.parameters['choice_sensitivity'])

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
        assert (retrieved_item_recall_probability == 0.0)
        assert(outcome_probability(cmr, 1) == 0.0)

    def test_stop_probability(self):
        cmr = experience(self.cmr)
        cmr = retrieve(cmr, 1)
        p_stop = stop_probability(cmr)
        assert(jnp.allclose(p_stop, 0.005033231))
        assert(jnp.allclose(outcome_probability(cmr, 0), 0.005033231))

    def test_outcome_probabilities(self):
        @jit
        def f():
            cmr = experience(self.cmr)
            cmr = retrieve(cmr, 1)
            return outcome_probabilities(cmr)
        p_all = f()
        desired_result = jnp.array([0.00503323, 0., 0.50872654, 0.101161, 0.0395892, 0.02791035, 0.02784999, 0.03489236, 0.0503365, 0.07826934, 0.12623152])
        assert(jnp.allclose(p_all.sum(), 1))
        assert(jnp.allclose(p_all, desired_result))