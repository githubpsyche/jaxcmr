import jax.numpy as jnp
from jax import random

from jaxcmr.simulation import (
    MemorySearchSimulator,
    item_to_study_positions,
    parameter_shifted_simulate_h5_from_h5,
    preallocate_for_h5_dataset,
    simulate_h5_from_h5,
)


class _FakeMemorySearch:
    """Minimal in-memory implementation of the MemorySearch protocol for tests."""

    def __init__(self, item_count: int = 3, active: bool = True):
        self.item_count = item_count
        self.is_active = jnp.array(active)

    # Study/retrieval transitions
    def experience(self, choice):  # noqa: D401 - protocol impl
        return self

    def start_retrieving(self):  # noqa: D401 - protocol impl
        return self

    def retrieve(self, choice):  # noqa: D401 - protocol impl
        # Stop retrieval when choice == 0, else remain active.
        self.is_active = jnp.array(False) if int(choice) == 0 else self.is_active
        return self

    # Probabilities API
    def activations(self):  # noqa: D401 - protocol impl
        return jnp.zeros((self.item_count,), dtype=jnp.float32)

    def outcome_probability(self, choice):  # noqa: D401 - protocol impl
        return jnp.array(1.0 if int(choice) == 0 else 0.0, dtype=jnp.float32)

    def outcome_probabilities(self):  # noqa: D401 - protocol impl
        # Always choose to stop (index 0)
        return jnp.array([1.0] + [0.0] * self.item_count, dtype=jnp.float32)


class _FakeModelFactory:
    """Minimal factory matching MemorySearchModelFactory for tests."""

    def __init__(self, dataset, connections):  # noqa: D401 - protocol impl
        self.dataset = dataset
        self.connections = connections

    def create_model(self, parameters):  # noqa: D401 - protocol impl
        return _FakeMemorySearch(item_count=int(self.dataset["pres_itemnos"].shape[1]))

    def create_trial_model(self, trial_index, parameters):  # noqa: D401 - impl
        return _FakeMemorySearch(item_count=int(self.dataset["pres_itemnos"].shape[1]))


class _IndexedMemorySearch(_FakeMemorySearch):
    """Fake memory search storing a recall index driven by per-subject parameters."""

    def __init__(self, item_count: int, recall_index: int):
        super().__init__(item_count=item_count)
        self.recall_index = recall_index


class _IndexedFactory:
    """Factory that encodes an integer recall index from the ``alpha`` parameter."""

    def __init__(self, dataset, connections):
        self.dataset = dataset
        self.connections = connections

    def create_model(self, parameters):
        item_count = int(self.dataset["pres_itemnos"].shape[1])
        return _IndexedMemorySearch(item_count, int(parameters["alpha"]))

    def create_trial_model(self, trial_index, parameters):
        item_count = int(self.dataset["pres_itemnos"].shape[1])
        return _IndexedMemorySearch(item_count, int(parameters["alpha"]))


def _recall_item_selected_by_parameter(model, present, trial, rng):
    """Returns a single recalled item chosen by the model's encoded index."""
    recall_item = present[int(model.recall_index)]
    return model, jnp.array([recall_item, 0], dtype=jnp.int32)


def test_returns_zeros_when_item_is_padding():
    """Behavior: Returns all zeros for padding item.

    Given:
      - item equals 0 and any presentation
    When:
      - converting to study positions with a fixed size
    Then:
      - the returned vector contains only zeros
    Why this matters:
      - invariant: 0 indicates no item and should not map to positions
    """
    # Arrange / Given
    present = jnp.array([3, 1, 2, 3])
    # Act / When
    out = item_to_study_positions(0, present, size=3)
    # Assert / Then
    assert out.shape == (3,)
    assert jnp.all(out == 0)


def test_returns_one_indexed_positions_when_item_repeats():
    """Behavior: Returns 1-indexed study positions where item occurs.

    Given:
      - a presentation with repeated occurrences of the same item
    When:
      - requesting up to three positions
    Then:
      - positions reflect 1-indexed locations with zero padding
    Why this matters:
      - invariant: downstream reindexing depends on consistent 1-indexing
    """
    # Arrange / Given
    present = jnp.array([5, 7, 5, 5])
    # Act / When
    out = item_to_study_positions(5, present, size=3)
    # Assert / Then
    assert out.tolist() == [1, 3, 4]


def test_reindexes_recalls_when_given_within_list_positions():
    """Behavior: Reindexes recalls from positions to presented item numbers.

    Given:
      - recalls that encode within-list positions (1-based)
    When:
      - initializing the MemorySearchSimulator
    Then:
      - stored trials are reindexed to the presented item numbers
    Why this matters:
      - invariant: simulator forwards item-number-coded recalls to trial function
    """
    # Arrange / Given
    dataset = {
        "subject": jnp.array([[0]]),
        "pres_itemnos": jnp.array([[10, 20, 30]]),
        "recalls": jnp.array([[2, 3, 0]]),  # positions -> should map to [20, 30, 0]
        "listLength": jnp.array([[3]]),
    }

    # Act / When
    sim = MemorySearchSimulator(_FakeModelFactory, dataset, connections=None)

    # Assert / Then
    assert sim.present_lists.shape == (1, 3)
    assert sim.trials.shape == (1, 3)
    assert jnp.all(sim.trials[0] == jnp.array([20, 30, 0]))


def test_sets_first_study_position_when_reindexing_after_simulation():
    """Behavior: Writes first study position of recalled item into output dataset.

    Given:
      - a single selected trial to simulate with two replications
    When:
      - simulating with a custom trial function that recalls an item-number
    Then:
      - output recalls contain the first 1-indexed study position of that item
    Why this matters:
      - invariant: downstream analyses expect recalls to be 1-indexed study positions
    """
    # Arrange / Given
    dataset = {
        "subject": jnp.array([[0]]),
        "pres_itemnos": jnp.array([[10, 20, 20]]),
        "recalls": jnp.array([[1, 0, 0]]),  # unused by custom trial fn
        "listLength": jnp.array([[3]]),
    }

    # Custom trial function that echoes a single recalled item-number (20)
    def echo_trial_fn(model, present, trial, rng):
        return model, jnp.array([20, 0, 0], dtype=jnp.int32)

    params = {"alpha": jnp.array([1.0])}
    mask = jnp.array([True])
    rng = random.PRNGKey(0)

    # Act / When
    out = simulate_h5_from_h5(
        _FakeModelFactory,
        dataset,
        connections=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=2,
        rng=rng,
        size=3,
        simulate_trial_fn=echo_trial_fn,
    )

    # Assert / Then
    # Two replications of the single selected trial
    assert out["subject"].shape[0] == 2
    # First study position of item-number 20 in [10, 20, 20] is 2
    assert out["recalls"].shape == (2, 3)
    assert int(out["recalls"][0, 0]) == 2


def test_simulate_h5_from_h5_maps_nonconsecutive_subject_ids():
    """Behavior: Aligns per-subject parameters using subject identifiers.

    Given:
      - a dataset whose subject ids are sparse and nonconsecutive
    When:
      - simulating trials with a factory that encodes subject-specific recall slots
    Then:
      - the recalls reflect the parameter row associated with each subject id
    Why this matters:
      - regression: HealeyKahana2014 simulations rely on sparse subject numbering
    """
    # Arrange / Given
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

    # Act / When
    out = simulate_h5_from_h5(
        _IndexedFactory,
        dataset,
        connections=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=1,
        rng=rng,
        size=2,
        simulate_trial_fn=_recall_item_selected_by_parameter,
    )

    # Assert / Then
    assert out["recalls"].shape == (2, 2)
    assert int(out["recalls"][0, 0]) == 1
    assert int(out["recalls"][1, 0]) == 2


def test_parameter_shifted_simulate_h5_from_h5_maps_subject_ids():
    """Behavior: Applies subject-id mapping while sweeping parameters.

    Given:
      - a dataset with a single nonconsecutive subject id and subject-keyed parameters
    When:
      - sweeping the ``alpha`` parameter across two values
    Then:
      - each simulated dataset reflects the swept parameter for that subject id
    Why this matters:
      - regression: parameter sweeps must stay aligned with sparse subject ids
    """
    # Arrange / Given
    dataset = {
        "subject": jnp.array([[138]], dtype=jnp.int32),
        "pres_itemnos": jnp.array([[30, 31]], dtype=jnp.int32),
        "recalls": jnp.zeros((1, 2), dtype=jnp.int32),
        "listLength": jnp.array([[2]], dtype=jnp.int32),
    }
    params = {
        "alpha": jnp.array([0], dtype=jnp.int32),
        "subject": jnp.array([138], dtype=jnp.int32),
    }
    mask = jnp.array([True])
    rng = random.PRNGKey(11)

    # Act / When
    sims = parameter_shifted_simulate_h5_from_h5(
        _IndexedFactory,
        dataset,
        connections=None,
        parameters=params,
        trial_mask=mask,
        experiment_count=1,
        varied_parameter="alpha",
        parameter_values=[0, 1],
        rng=rng,
        size=2,
        simulate_trial_fn=_recall_item_selected_by_parameter,
    )

    # Assert / Then
    assert len(sims) == 2
    assert int(sims[0]["recalls"][0, 0]) == 1
    assert int(sims[1]["recalls"][0, 0]) == 2


def test_replicates_selected_trials_when_preallocating_dataset():
    """Behavior: Replicates masked trials by experiment_count in all arrays.

    Given:
      - a dataset with two trials and a mask selecting one
    When:
      - preallocating with experiment_count set to two
    Then:
      - the selected trial appears twice across all fields
    Why this matters:
      - invariant: simulation allocates output rows based on trial replication
    """
    # Arrange / Given
    data = {
        "subject": jnp.array([[0], [1]]),
        "pres_itemnos": jnp.array([[1, 2, 3], [4, 5, 6]]),
        "recalls": jnp.array([[1, 0, 0], [2, 0, 0]]),
        "listLength": jnp.array([[3], [3]]),
    }
    mask = jnp.array([True, False])

    # Act / When
    out = preallocate_for_h5_dataset(data, mask, experiment_count=2)

    # Assert / Then
    assert out["subject"].shape[0] == 2
    assert jnp.all(out["pres_itemnos"] == jnp.array([[1, 2, 3], [1, 2, 3]]))
    assert jnp.all(out["recalls"] == jnp.array([[1, 0, 0], [1, 0, 0]]))
