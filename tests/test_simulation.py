import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from support import CMR_PARAMS
from jaxcmr.models.cmr import CMR
from jaxcmr.components.response import RecallUntilExhausted
from jaxcmr.dataset import make_dataset
from jaxcmr.helpers import save_dict_to_hdf5
from jaxcmr.simulation import (
    MemorySearchSimulator,
    simulate_study,
    simulate_dataset,
    simulate_parameter_sweep,
    load_matching_simulation,
)


class _ParameterFactory:
    """Only the trial constructor and value used by the supplied simulator."""

    def __init__(self, dataset, features):
        pass

    def create_trial_model(self, trial_index, parameters):
        return SimpleNamespace(recall_index=parameters['alpha'])


def _recall_item_selected_by_parameter(model, present, trial, rng):
    """Returns a single recalled item chosen by the model's encoded index."""
    recall_item = present[model.recall_index]
    return (model, jnp.array([recall_item, 0], dtype=jnp.int32))


def test_study_matches_replayed_events_before_retrieval():
    model = CMR(4, CMR_PARAMS, response_policy_create_fn=RecallUntilExhausted)
    present = jnp.array([2, 1, 2, 3])
    initial = [np.array(value, copy=True) for value in jax.tree.leaves(model)]
    expected = model
    for item in present:
        expected = expected.experience(item)
    actual = jax.jit(simulate_study)(model, present)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=1e-7)
    for observed, reference in zip(jax.tree.leaves(model), initial, strict=True):
        np.testing.assert_array_equal(observed, reference)


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
    out = simulate_dataset(
        _ParameterFactory,
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


def test_simulate_dataset_maps_nonconsecutive_subject_ids():
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
    out = simulate_dataset(
        _ParameterFactory,
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
    assert int(out["recalls"][1, 0]) == 2, "Simulation must assign parameters by subject identity"

    # Reuse the same prepared dataset for A/B/A parameter assignments.
    simulator = MemorySearchSimulator(
        _ParameterFactory, dataset, None, _recall_item_selected_by_parameter
    )
    original = deepcopy(dataset)
    for alpha, expected in [([0, 1], [[1, 0], [2, 0]]),
                            ([1, 0], [[2, 0], [1, 0]]),
                            ([0, 1], [[1, 0], [2, 0]])]:
        result = simulator.simulate({**params, "alpha": jnp.array(alpha)}, rng, size=2)
        np.testing.assert_array_equal(result["recalls"], expected)
    for field, values in dataset.items():
        np.testing.assert_array_equal(values, original[field])

    def simulate_parameters(alpha):
        return simulate_dataset(
            _ParameterFactory, dataset, None, {**params, 'alpha': alpha}, mask, 1, rng,
            size=2, simulate_trial_fn=_recall_item_selected_by_parameter,
        )['recalls']

    mapped = jax.vmap(simulate_parameters)(jnp.array([[0, 1], [1, 0]]))
    np.testing.assert_array_equal(mapped, [[[1, 0], [2, 0]], [[2, 0], [1, 0]]])


def test_parameter_sweep_maps_subject_ids():
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
    sims = simulate_parameter_sweep(
        _ParameterFactory,
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


def test_load_matching_simulation_settings_and_legacy_metadata(tmp_path):
    path = tmp_path / "responses.h5"
    data = make_dataset(recalls=[1, 0], listLength=2)
    save_dict_to_hdf5(data, path)
    saved = {"implementation": {"jaxcmr_version": "v", "jax_enable_x64": True},
             "data_path": "data/example.h5", "seed": 1,
             "provenance": {"identity": "obsolete"}, "data_sha256": "obsolete"}
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(saved))
    expected = {"jaxcmr_version": "v", "jax_enable_x64": True,
                "data_path": "data/example.h5", "seed": 1}
    before = deepcopy(expected), path.read_bytes(), metadata_path.read_bytes()
    loaded = load_matching_simulation(str(path), expected)
    assert loaded is not None
    assert loaded.keys() == data.keys()
    for field, values in data.items():
        np.testing.assert_array_equal(loaded[field], values)
    for field, value in [("seed", 2), ("data_path", "other.h5"),
                         ("missing", 1), ("jaxcmr_version", "new")]:
        assert load_matching_simulation(path, {**expected, field: value}) is None
    assert (expected, path.read_bytes(), metadata_path.read_bytes()) == before


def test_load_matching_simulation_distinguishes_missing_fields_from_none(tmp_path):
    path = tmp_path / "responses.h5"
    save_dict_to_hdf5(make_dataset(recalls=[1, 0], listLength=2), path)
    metadata_path = path.with_suffix(".json")
    for field in ("feature_column", "jaxcmr_version", "jax_enable_x64"):
        expected = {field: None}
        metadata_path.write_text("{}")
        assert load_matching_simulation(path, expected) is None
        metadata_path.write_text(json.dumps(expected))
        assert load_matching_simulation(path, expected) is not None

    metadata_path.write_text(json.dumps({"implementation": {"jaxcmr_version": None}}))
    assert load_matching_simulation(path, {"jaxcmr_version": None}) is not None
    metadata_path.write_text(json.dumps({
        "implementation": {"jaxcmr_version": None}, "jaxcmr_version": "current"
    }))
    assert load_matching_simulation(path, {"jaxcmr_version": "current"}) is not None
    assert load_matching_simulation(path, {"jaxcmr_version": None}) is None


def test_load_matching_simulation_missing_files_and_read_errors(tmp_path):
    path = tmp_path / "responses.h5"
    metadata_path = path.with_suffix(".json")
    expected = {"seed": 1}
    assert load_matching_simulation(path, expected) is None
    metadata_path.write_text("not JSON")
    assert load_matching_simulation(path, expected) is None
    metadata_path.unlink()
    path.write_bytes(b"not HDF5")
    assert load_matching_simulation(path, expected) is None
    metadata_path.write_text("not JSON")
    with pytest.raises(json.JSONDecodeError):
        load_matching_simulation(path, expected)
    metadata_path.write_text(json.dumps({"seed": 2}))
    assert load_matching_simulation(path, expected) is None
    metadata_path.write_text(json.dumps(expected))
    with pytest.raises(OSError):
        load_matching_simulation(path, expected)
