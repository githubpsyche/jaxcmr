import jax.numpy as jnp
import pytest
from jaxcmr.helpers import has_repeats_per_row
from jaxcmr.repetition import make_control_dataset


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
