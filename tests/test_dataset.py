"""Condition selections and participant coverage on trial-indexed data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcmr.dataset import TrialSelection, apply_by_subject, make_dataset, select_conditions


def sample_data():
    return dict(recalls=jnp.array([[1,0],[2,0],[1,2],[0,0]]),
                subject=jnp.array([[8],[3],[8],[12]]),
                pres_itemnos=jnp.array([[1,2],[1,0],[1,2],[1,2]]),
                condition=jnp.array([1,1,2,2]))


def test_select_query_masks_filtering_order_and_nonmutation():
    data=sample_data();before={k:np.asarray(v).copy() for k,v in data.items()}
    result=select_conditions(data,["data['condition'] == 1",data['condition']==2,None],complete_lists_only=True)
    assert all(isinstance(s,TrialSelection) for s in result)
    np.testing.assert_array_equal(result[0].mask,[True,False,False,False])
    assert [s.trial_count for s in result]==[1,2,3]
    assert [s.participant_count for s in result]==[1,2,2]
    np.testing.assert_array_equal(result[2].subject_ids,[8,12])
    np.testing.assert_array_equal(select_conditions(data,[data['condition']==1])[0].mask,
                                  select_conditions(data,["data['condition'] == 1"])[0].mask)
    for k in data:np.testing.assert_array_equal(data[k],before[k])
    empty=select_conditions(data,[jnp.zeros(4,bool)],require_nonempty=False)[0]
    assert empty.trial_count==empty.participant_count==0
    with pytest.raises(ValueError,match='no eligible'):
        select_conditions(data,[jnp.zeros(4,bool)])


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
    np.testing.assert_array_equal(mapped['recalls'][:, 0], recalls)
    np.testing.assert_array_equal(mapped['pres_itemnos'][:, 0], presentations)
    generated = jax.jit(lambda r: make_dataset(r, listLength=3))(recalls)
    np.testing.assert_array_equal(generated['pres_itemnos'], [[1, 2, 3]] * 2)


def test_apply_by_subject_preserves_trial_fields():
    data = make_dataset(recalls=jnp.array([[1, 0], [2, 0]]), subject=jnp.array([[1], [2]]))
    data["condition"] = jnp.array([[10], [20]])
    selected = apply_by_subject(data, jnp.array([True, True]), lambda subject: subject["condition"])
    assert [int(values[0, 0]) for values in selected] == [10, 20]
