"""Condition selections and participant coverage on trial-indexed data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.dataset import (
    TrialSelection,
    apply_by_subject,
    generate_recall_mask,
    generate_trial_mask,
    limit_to_first_subjects,
    make_dataset,
    make_subject_trial_masks,
    replace_recalls,
    select_conditions,
    select_trials,
)


def sample_data():
    return {
        "recalls": jnp.array([[1, 0], [2, 0], [1, 2], [0, 0]]),
        "subject": jnp.array([[8], [3], [8], [12]]),
        "pres_itemnos": jnp.array([[1, 2], [1, 0], [1, 2], [1, 2]]),
        "condition": jnp.array([1, 1, 2, 2]),
    }


def test_select_query_masks_filtering_order_and_nonmutation():
    data = sample_data()
    before = {k: np.asarray(v).copy() for k, v in data.items()}
    result = select_conditions(
        data,
        ["data['condition'] == 1", data["condition"] == 2, None],
        complete_lists_only=True,
    )
    assert all(isinstance(s, TrialSelection) for s in result)
    np.testing.assert_array_equal(result[0].mask, [True, False, False, False])
    assert [s.trial_count for s in result] == [1, 2, 3]
    assert [s.participant_count for s in result] == [1, 2, 2]
    np.testing.assert_array_equal(result[2].subject_ids, [8, 12])
    np.testing.assert_array_equal(
        select_conditions(data, [data["condition"] == 1])[0].mask,
        select_conditions(data, ["data['condition'] == 1"])[0].mask,
    )
    for k in data:
        np.testing.assert_array_equal(data[k], before[k])
    query_mask = generate_trial_mask(data, "np.array([True, False, True, False])")
    assert isinstance(query_mask, jax.Array)
    np.testing.assert_array_equal(query_mask, [True, False, True, False])
    recall_mask = generate_recall_mask(data, "np.asarray(data['recalls']) > 0")
    assert isinstance(recall_mask, jax.Array)
    np.testing.assert_array_equal(recall_mask, data["recalls"] > 0)
    empty = select_conditions(data, [jnp.zeros(4, bool)], require_nonempty=False)[0]
    assert empty.trial_count == empty.participant_count == 0
    with pytest.raises(ValueError, match="no eligible"):
        select_conditions(data, [jnp.zeros(4, bool)])


def test_constructor_broadcasts_then_repeats_and_pads_mixed_lengths():
    data = make_dataset([1, 0], listLength=[2, 3], subject=[7, 9], reps=2)
    np.testing.assert_array_equal(data["subject"].ravel(), [7, 9, 7, 9])
    np.testing.assert_array_equal(data["pres_itemnos"], [[1, 2, 0], [1, 2, 3]] * 2)
    np.testing.assert_array_equal(data["recalls"], [[1, 0]] * 4)
    np.testing.assert_array_equal(data["listLength"].ravel(), [2, 3, 2, 3])
    padded = make_dataset([1, 0], [1, 2, 0, 0], listLength=2)
    assert int(padded["listLength"][0, 0]) == 2
    # Width inference remains the compatibility default, not an experimental claim.
    assert int(make_dataset([1, 0], [1, 2, 0, 0])["listLength"][0, 0]) == 4
    assert int(make_dataset([3, 0])["listLength"][0, 0]) == 3


def test_constructor_composes_with_jit_and_vmap():
    recalls = jnp.array([[1, 0, 0], [2, 1, 0]])
    presentations = jnp.array([[1, 2, 0], [1, 2, 3]])
    lengths, subjects = jnp.array([2, 3]), jnp.array([7, 9])
    build = lambda r, p, n, s: make_dataset(r, p, n, s, reps=2)
    expected = build(recalls, presentations, lengths, subjects)
    actual = jax.jit(build)(recalls, presentations, lengths, subjects)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    mapped = jax.vmap(lambda r, p, n, s: make_dataset(r, p, n, s))(
        recalls, presentations, lengths, subjects
    )
    np.testing.assert_array_equal(mapped["recalls"][:, 0], recalls)
    np.testing.assert_array_equal(mapped["pres_itemnos"][:, 0], presentations)
    generated = jax.jit(lambda r: make_dataset(r, listLength=3))(recalls)
    np.testing.assert_array_equal(generated["pres_itemnos"], [[1, 2, 3]] * 2)


def test_apply_by_subject_preserves_trial_fields():
    data = make_dataset([[1, 0], [2, 0], [1, 2], [0, 0]], subject=[8, 3, 8, 12])
    data["condition"] = jnp.array([[10], [20], [30], [40]])
    original = {name: np.asarray(values).copy() for name, values in data.items()}
    trial_mask = jnp.array([True, True, False, False])
    masks, subjects = make_subject_trial_masks(trial_mask, data["subject"].ravel())
    np.testing.assert_array_equal(subjects, [3, 8])
    np.testing.assert_array_equal(
        masks, [[False, True, False, False], [True, False, False, False]]
    )
    selected = apply_by_subject(
        data,
        trial_mask,
        lambda subject, *, scale: scale * subject["condition"],
        scale=2,
    )
    assert [values.tolist() for values in selected] == [[[40]], [[20]]]
    empty_mask = jnp.zeros(4, dtype=bool)
    empty_masks, empty_subjects = make_subject_trial_masks(
        empty_mask, data["subject"].ravel()
    )
    assert empty_masks == [] and empty_subjects.size == 0
    assert (
        apply_by_subject(data, empty_mask, lambda subject: subject["condition"]) == []
    )
    for name, values in original.items():
        np.testing.assert_array_equal(data[name], values)


def test_trial_selection_and_recall_replacement_preserve_study_data():
    data = make_dataset([1, 0], [[1, 2], [2, 1], [3, 1]], subject=[8, 3, 8])
    data["condition"] = jnp.array([10, 20, 30])
    data["measurement"] = jnp.arange(12).reshape(3, 2, 2)
    data["irt"] = jnp.ones((3, 2))
    data["response_confidence"] = jnp.zeros((3, 2))
    original = {name: np.asarray(values).copy() for name, values in data.items()}
    selected = jax.jit(select_trials)(data, jnp.array([1, 0, 1]))
    np.testing.assert_array_equal(selected["subject"].ravel(), [3, 8, 3])
    np.testing.assert_array_equal(selected["pres_itemnos"], [[2, 1], [1, 2], [2, 1]])
    np.testing.assert_array_equal(selected["condition"], [20, 10, 20])
    np.testing.assert_array_equal(
        selected["measurement"],
        [
            original["measurement"][1],
            original["measurement"][0],
            original["measurement"][1],
        ],
    )
    replacement = replace_recalls(
        selected, jnp.array([[2], [1], [2]]), ["response_confidence"]
    )
    np.testing.assert_array_equal(replacement["recalls"], [[2], [1], [2]])
    assert "irt" not in replacement and "response_confidence" not in replacement
    assert "irt" in selected and "response_confidence" in selected
    for name in ["subject", "pres_itemnos", "listLength", "condition", "measurement"]:
        np.testing.assert_array_equal(replacement[name], selected[name])
    for name, values in original.items():
        np.testing.assert_array_equal(data[name], values)


def test_participant_limit_keeps_original_rows_and_handles_empty_input():
    data = {
        "subject": jnp.array([[8], [3], [8], [12]]),
        "condition": jnp.array([10, 20, 30, 40]),
    }
    limited = limit_to_first_subjects(data, 2)
    np.testing.assert_array_equal(limited["subject"].ravel(), [8, 3, 8])
    np.testing.assert_array_equal(limited["condition"], [10, 20, 30])
    for limit in [0, 99]:
        result = limit_to_first_subjects(data, limit)
        assert result is not data
        for name, values in data.items():
            np.testing.assert_array_equal(result[name], values)
    empty = select_trials(make_dataset([1, 0], [1, 2]), jnp.array([], dtype=int))
    for limit in [0, 2]:
        result = limit_to_first_subjects(empty, limit)
        assert {name: values.shape for name, values in result.items()} == {
            "recalls": (0, 2),
            "pres_itemnos": (0, 2),
            "listLength": (0, 1),
            "subject": (0, 1),
        }
