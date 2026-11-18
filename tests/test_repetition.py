import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.repetition import (
    equivalent_study_positions,
    filter_repeated_recalls,
    has_repeats_per_row,
    item_to_study_positions,
    make_control_dataset,
    recalls_to_first_study_positions,
)


def _toy_dataset_for_controls():
    L = 4
    subject = jnp.array([[1], [1], [2], [2]], dtype=jnp.int32)
    listLength = jnp.array([[L], [L], [L], [L]], dtype=jnp.int32)
    pres_itemnos = jnp.array(
        [[1, 2, 3, 4], [1, 2, 1, 4], [1, 2, 3, 4], [1, 2, 2, 3]], dtype=jnp.int32
    )
    recalls = jnp.array([[1, 2, 3, 0], [3, 1, 1, 0], [1, 1, 2, 0], [2, 3, 0, 0]], dtype=jnp.int32)
    list_type = jnp.array([[0], [1], [0], [1]], dtype=jnp.int32)
    return {
        "subject": subject,
        "listLength": listLength,
        "pres_itemnos": pres_itemnos,
        "recalls": recalls,
        "list_type": list_type,
    }


@pytest.mark.parametrize("remove", [True, False])
def test_toggles_repeated_recalls_removal(remove: bool):
    """``remove_repeats`` controls whether within-trial repeats are zeroed."""
    data = _toy_dataset_for_controls()
    ctrl = make_control_dataset(
        data=data,
        mixed_query="data['list_type'].flatten() == 1",
        control_query="data['list_type'].flatten() == 0",
        n_shuffles=2,
        remove_repeats=remove,
        seed=0,
    )
    has_repeats = has_repeats_per_row(ctrl["recalls"]).any().item()
    assert has_repeats is (not remove)


def test_control_preserves_study_fields_but_drops_old_recall_metadata():
    data = _toy_dataset_for_controls()
    data["layout_id"] = jnp.arange(4)[:, None]
    data["rec_itemids"] = jnp.ones_like(data["recalls"])
    data["irt"] = jnp.ones_like(data["recalls"])
    data["cue_clips"] = jnp.ones_like(data["recalls"])
    data["_should_tabulate"] = jnp.ones_like(data["recalls"], dtype=bool)
    data["custom_recall_field"] = jnp.ones_like(data["recalls"])
    control = make_control_dataset(
        data,
        mixed_query="data['list_type'].flatten() == 1",
        control_query="data['list_type'].flatten() == 0",
        n_shuffles=1,
        recall_dependent_fields={"custom_recall_field"},
    )
    assert "listLength" in control
    assert "layout_id" in control
    assert set(control).isdisjoint(
        {"rec_itemids", "irt", "cue_clips", "_should_tabulate", "custom_recall_field"}
    )
    assert jnp.array_equal(control["layout_id"].flatten(), jnp.array([1, 3]))


def test_control_infers_list_length_for_legacy_dataset():
    data = _toy_dataset_for_controls()
    data.pop("listLength")
    control = make_control_dataset(
        data,
        mixed_query="data['list_type'].flatten() == 1",
        control_query="data['list_type'].flatten() == 0",
        n_shuffles=1,
    )
    assert jnp.array_equal(control["listLength"], jnp.array([[4], [4]]))


def test_occurrence_operations_compose_without_compacting_recalls():
    presentation = jnp.array([10, 20, 10, 30])
    np.testing.assert_array_equal(
        jax.jit(item_to_study_positions, static_argnames='size')(10, presentation, 3),
        [1, 3, 0],
    )
    np.testing.assert_array_equal(item_to_study_positions(10, presentation, 1), [1])
    positions = jax.jit(jax.vmap(equivalent_study_positions, in_axes=(0, None, None)),
                        static_argnums=2)(jnp.array([3, 2, 0]), presentation, 3)
    np.testing.assert_array_equal(positions, [[1, 3, 0], [2, 0, 0], [0, 0, 0]])
    recalls = jnp.array([3, 2, 1, 4, 0])
    canonical = jax.jit(recalls_to_first_study_positions)(recalls, presentation)
    np.testing.assert_array_equal(canonical, [1, 2, 1, 4, 0])
    np.testing.assert_array_equal(filter_repeated_recalls(recalls[None]), recalls[None])
    np.testing.assert_array_equal(
        jax.jit(filter_repeated_recalls)(canonical[None]), [[1, 2, 0, 4, 0]]
    )
    np.testing.assert_array_equal(
        has_repeats_per_row(jnp.array([[0, 0, -1, -1], [3, 2, 3, 0]])), [False, True]
    )
    np.testing.assert_array_equal(recalls, [3, 2, 1, 4, 0])


@pytest.mark.parametrize('pure_count,mixed_count', [(3, 5), (5, 3), (3, 3)])
def test_each_shuffle_covers_every_mixed_trial_with_participant_specific_pure_rows(
    pure_count, mixed_count
):
    trial_count = pure_count + mixed_count
    shuffle_count = 3
    data = {
        'subject': jnp.repeat(jnp.array([7, 19]), trial_count)[:, None],
        'listLength': jnp.full((2 * trial_count, 1), 12),
        'pres_itemnos': jnp.array(
            ([list(range(1, 13))] * pure_count + [list(range(1, 12)) + [11]] * mixed_count) * 2
        ),
        'recalls': jnp.array([
            [position, 0, 0]
            for offset in (0, 5)
            for position in list(range(1 + offset, pure_count + 1 + offset)) + [0] * mixed_count
        ]),
        'condition': jnp.array(([0] * pure_count + [1] * mixed_count) * 2),
        'layout_id': jnp.arange(2 * trial_count),
        'study_measure': jnp.arange(4 * trial_count).reshape(2 * trial_count, 2),
    }
    before = {name: np.asarray(values).copy() for name, values in data.items()}
    controls = make_control_dataset(
        data, "data['condition'] == 1", "data['condition'] == 0", shuffle_count, seed=9
    )
    again = make_control_dataset(
        data, "data['condition'] == 1", "data['condition'] == 0", shuffle_count, seed=9
    )
    for participant_index, offset in enumerate((0, 5)):
        block_start = participant_index * mixed_count * shuffle_count
        for shuffle in range(shuffle_count):
            rows = slice(block_start + shuffle * mixed_count, block_start + (shuffle + 1) * mixed_count)
            expected_indices = np.arange(pure_count, trial_count) + participant_index * trial_count
            for field in ('subject', 'listLength', 'pres_itemnos', 'condition', 'layout_id', 'study_measure'):
                np.testing.assert_array_equal(controls[field][rows], before[field][expected_indices])
            selected = np.asarray(controls['recalls'][rows, 0])
            available = set(range(1 + offset, pure_count + 1 + offset))
            for start in range(0, mixed_count, pure_count):
                permutation = selected[start:start + pure_count]
                assert len(set(permutation)) == len(permutation)
                assert set(permutation) <= available
                if len(permutation) == pure_count:
                    assert set(permutation) == available
    assert controls['recalls'].shape == (2 * shuffle_count * mixed_count, 3)
    for field in controls:
        np.testing.assert_array_equal(controls[field], again[field])
    for field, values in data.items():
        np.testing.assert_array_equal(values, before[field])
