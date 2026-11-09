import jax.numpy as jnp
from jax import random
from jaxcmr.simulation import parameter_shifted_simulate_h5_from_h5, simulate_h5_from_h5


class _FakeContext:
    """Minimal context satisfying the Context protocol."""

    def __init__(self, size: int = 3):
        self.size = size
        self.state = jnp.zeros(size)
        self.initial_state = jnp.zeros(size)

    def integrate(self, context_input, drift_rate):
        return self


class _FakeMemorySearch:
    """Minimal in-memory implementation of the MemorySearch protocol for tests."""

    def __init__(self, item_count: int = 3, active: bool = True):
        self.item_count = item_count
        self.is_active = jnp.array(active)
        self.recallable = jnp.ones(item_count, dtype=bool)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(item_count, dtype=int)
        self.context = _FakeContext(item_count)

    def experience(self, choice):
        return self

    def start_retrieving(self):
        return self

    def retrieve(self, choice):
        self.is_active = jnp.array(False) if int(choice) == 0 else self.is_active
        return self

    def activations(self):
        return jnp.zeros((self.item_count,), dtype=jnp.float32)

    def outcome_probability(self, choice):
        return jnp.array(1.0 if int(choice) == 0 else 0.0, dtype=jnp.float32)

    def outcome_probabilities(self):
        return jnp.array([1.0] + [0.0] * self.item_count, dtype=jnp.float32)


class _FakeModelFactory:
    """Minimal factory matching MemorySearchModelFactory for tests."""

    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def create_model(self, parameters):
        return _FakeMemorySearch(item_count=int(self.dataset["pres_itemnos"].shape[1]))

    def create_trial_model(self, trial_index, parameters):
        return _FakeMemorySearch(item_count=int(self.dataset["pres_itemnos"].shape[1]))


class _IndexedMemorySearch(_FakeMemorySearch):
    """Fake memory search storing a recall index driven by per-subject parameters."""

    def __init__(self, item_count: int, recall_index: int):
        super().__init__(item_count=item_count)
        self.recall_index = recall_index


class _IndexedFactory:
    """Factory that encodes an integer recall index from the ``alpha`` parameter."""

    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def create_model(self, parameters):
        item_count = int(self.dataset["pres_itemnos"].shape[1])
        return _IndexedMemorySearch(item_count, parameters["alpha"])

    def create_trial_model(self, trial_index, parameters):
        item_count = int(self.dataset["pres_itemnos"].shape[1])
        return _IndexedMemorySearch(item_count, parameters["alpha"])


def _recall_item_selected_by_parameter(model, present, trial, rng):
    """Returns a single recalled item chosen by the model's encoded index."""
    recall_item = present[model.recall_index]
    return (model, jnp.array([recall_item, 0], dtype=jnp.int32))


def test_sets_first_study_position_when_reindexing_after_simulation():
    """Writes first study position of recalled item into output dataset."""
    dataset = {
        "subject": jnp.array([[0]]),
        "pres_itemnos": jnp.array([[10, 20, 20]]),
        "recalls": jnp.array([[1, 0, 0]]),
        "listLength": jnp.array([[3]]),
        "condition": jnp.array([[1, 2, 2]]),
        "rec_itemids": jnp.array([[10, 0, 0]]),
        "irt": jnp.array([[100, 0, 0]]),
        "cue_clips": jnp.array([[10, 0, 0]]),
        "_should_tabulate": jnp.array([[True, False, False]]),
        "custom_recall_field": jnp.array([[1, 0, 0]]),
    }

    def echo_trial_fn(model, present, trial, rng):
        return (model, jnp.array([20, 0, 0], dtype=jnp.int32))

    params = {"alpha": jnp.array([1.0])}
    mask = jnp.array([True])
    rng = random.PRNGKey(0)
    out = simulate_h5_from_h5(
        _FakeModelFactory,
        dataset,
        features=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=2,
        rng=rng,
        size=3,
        simulate_trial_fn=echo_trial_fn,
        recall_dependent_fields={"custom_recall_field"},
    )
    assert out["subject"].shape[0] == 2
    assert out["recalls"].shape == (2, 3)
    assert int(out["recalls"][0, 0]) == 2
    assert "condition" in out
    assert set(out).isdisjoint(
        {"rec_itemids", "irt", "cue_clips", "_should_tabulate", "custom_recall_field"}
    )


def test_simulate_h5_from_h5_maps_nonconsecutive_subject_ids():
    """Aligns per-subject parameters using subject identifiers."""
    dataset = {
        "subject": jnp.array([[63], [138]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[10, 11], [20, 21]], dtype=jnp.int32),
        "recalls": jnp.zeros((2, 2), dtype=jnp.int32),
        "listLength": jnp.array([[2], [2]], dtype=jnp.int32),
    }
    params = {
        "alpha": jnp.array([0, 1], dtype=jnp.int32),
        "subject": jnp.array([63, 138], dtype=jnp.int32),
    }
    mask = jnp.array([True, True])
    rng = random.PRNGKey(10)
    out = simulate_h5_from_h5(
        _IndexedFactory,
        dataset,
        features=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=1,
        rng=rng,
        size=2,
        simulate_trial_fn=_recall_item_selected_by_parameter,
    )
    assert out["recalls"].shape == (2, 2)
    assert int(out["recalls"][0, 0]) == 1
    assert int(out["recalls"][1, 0]) == 2


def test_parameter_shifted_simulate_h5_from_h5_maps_subject_ids():
    """Applies subject-id mapping while sweeping parameters."""
    dataset = {
        "subject": jnp.array([[138]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[30, 31]], dtype=jnp.int32),
        "recalls": jnp.zeros((1, 2), dtype=jnp.int32),
        "listLength": jnp.array([[2]], dtype=jnp.int32),
        "rec_itemids": jnp.array([[30, 0]], dtype=jnp.int32),
    }
    params = {
        "alpha": jnp.array([0], dtype=jnp.int32),
        "subject": jnp.array([138], dtype=jnp.int32),
    }
    mask = jnp.array([True])
    rng = random.PRNGKey(11)
    sims = parameter_shifted_simulate_h5_from_h5(
        _IndexedFactory,
        dataset,
        features=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=1,
        varied_parameter="alpha",
        parameter_values=[0, 1],
        rng=rng,
        size=2,
        simulate_trial_fn=_recall_item_selected_by_parameter,
    )
    assert len(sims) == 2
    assert int(sims[0]["recalls"][0, 0]) == 1
    assert int(sims[1]["recalls"][0, 0]) == 2
    assert all(("rec_itemids" not in sim for sim in sims))
