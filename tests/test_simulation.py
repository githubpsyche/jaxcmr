from copy import deepcopy
from types import SimpleNamespace
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxcmr.simulation import (
    parameter_shifted_simulate_h5_from_h5,
    simulate_h5_from_h5,
    simulation_metadata_matches,
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

    def simulate_parameters(alpha):
        return simulate_h5_from_h5(
            _ParameterFactory, dataset, None, {**params, 'alpha': alpha}, mask, 1, rng,
            size=2, simulate_trial_fn=_recall_item_selected_by_parameter,
        )['recalls']

    mapped = jax.vmap(simulate_parameters)(jnp.array([[0, 1], [1, 0]]))
    np.testing.assert_array_equal(mapped, [[[1, 0], [2, 0]], [[2, 0], [1, 0]]])

    for invalid in (
        {"subject": jnp.array([63, 63]), "alpha": jnp.array([0, 1])},
        {"subject": jnp.array([63, 139]), "alpha": jnp.array([0, 1])},
        {"alpha": jnp.array([0, 1])},  # IDs 63/138 cannot be positional row indices.
    ):
        with pytest.raises(ValueError):
            simulate_h5_from_h5(_ParameterFactory, dataset, None, invalid, mask, 1, rng,
                               size=2, simulate_trial_fn=_recall_item_selected_by_parameter)


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


def test_metadata_legacy_versions_and_requested_settings():
    saved = {"implementation": {"jaxcmr_version": "v", "jax_enable_x64": True},
             "data_path": "data/example.h5", "seed": 1,
             "provenance": {"identity": "obsolete"}, "data_sha256": "obsolete"}
    expected = {"jaxcmr_version": "v", "jax_enable_x64": True,
                "data_path": "data/example.h5", "seed": 1}
    before = deepcopy((saved, expected))
    assert simulation_metadata_matches(saved, expected)
    assert (saved, expected) == before
    for field, value in [("seed", 2), ("data_path", "other.h5"),
                         ("missing", 1), ("jaxcmr_version", "new")]:
        assert not simulation_metadata_matches(saved, {**expected, field: value})

def test_metadata_distinguishes_missing_fields_from_none():
    # Cover ordinary fields and the two fields with a legacy fallback.
    for field in ("feature_column", "jaxcmr_version", "jax_enable_x64"):
        expected = {field: None}
        assert not simulation_metadata_matches({}, expected)
        assert simulation_metadata_matches({field: None}, expected)

    # An explicitly recorded legacy None is different from an absent field.
    saved = {"implementation": {"jaxcmr_version": None}}
    assert simulation_metadata_matches(saved, {"jaxcmr_version": None})
    # Current fields take precedence over conflicting legacy values.
    saved["jaxcmr_version"] = "current"
    assert simulation_metadata_matches(saved, {"jaxcmr_version": "current"})
    assert not simulation_metadata_matches(saved, {"jaxcmr_version": None})

