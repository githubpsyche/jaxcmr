"""
"""

# %% Imports

from jaxcmr._analyses import spc, _map_nonzero_recall_to_all_positions, map_recall_to_all_positions, map_single_trial_to_positions, item_occurrences, pfr
from jaxcmr.datasets import load_data, generate_trial_mask
from jaxcmr.analyses import (
    single_pres_spc,
    multi_pres_spc,
    single_pres_pfr,
    multi_pres_pfr,
)
import pytest
import numpy as np
from jax import numpy as jnp

# %% Fixtures

data_path = "D:/data/{}.h5"
param_path = "D:/data/results/{}_{}_{}.jsonl"


def recalls(data_tag, trial_query):
    data = load_data(data_path.format(data_tag))
    trial_mask = generate_trial_mask(data, trial_query)
    return data["recalls"][trial_mask]


def pres_itemnos(data_tag, trial_query):
    data = load_data(data_path.format(data_tag))
    trial_mask = generate_trial_mask(data, trial_query)
    return data["pres_itemnos"][trial_mask]


# %% Tests


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("HealyKahana2014", "data['listtype'] == -1"),
        ("LohnasKahana2014", "data['list_type'] == 1"),
    ],
)
def test_numba_spc(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    single_pres_result = single_pres_spc(trials, list_length)
    multi_pres_result = multi_pres_spc(trials, presentations, list_length)

    assert np.allclose(single_pres_result, multi_pres_result)


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("LohnasKahana2014", "data['list_type'] == 2"),
        ("LohnasKahana2014", "data['list_type'] == 3"),
        ("LohnasKahana2014", "data['list_type'] == 4"),
    ],
)
def test_single_and_multipres_spc_are_different(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    single_pres_result = single_pres_spc(trials, list_length)
    multi_pres_result = multi_pres_spc(trials, presentations, list_length)

    assert not np.allclose(single_pres_result, multi_pres_result)


def test_item_occurrences_item_repetitions():
    presentation = jnp.array([1, 2, 2, 3, 1, 3])
    trial = jnp.array([1, 3])
    result = item_occurrences(trial, presentation)
    assert jnp.all(result == jnp.array([1, 0, 0, 1, 1, 1]))


def test_empty_trials():
    """Validates the function's behavior when all the trials are empty."""
    trials = jnp.array([[], []])
    list_length = 3
    result = spc(trials, list_length)
    assert jnp.all(result == jnp.array([0.0, 0.0, 0.0]))


def test_all_present_trials():
    """Checks the case where all possible indices are present in every trial."""
    trials = jnp.array([[1, 2, 3], [1, 2, 3]])
    list_length = 3
    result = spc(trials, list_length)
    assert jnp.all(result == jnp.array([1.0, 1.0, 1.0]))


def test_multitrial_item_repetitions():
    presentations = jnp.array([[1, 2, 2, 3, 1, 3], [1, 2, 3, 4, 5, 6]])
    trials = jnp.array([[1, 3, 0,0,0,0], [3, 1, 5, 2, 0, 0]])
    result = spc(trials, presentations)
    assert jnp.all(result == jnp.array([1., .5, .5, .5, 1., .5]))


def test_map_recall_to_positions():
    presentation = jnp.array([1, 2, 2, 3, 1, 3])
    trial = jnp.array([1, 3, 0,0,0,0])
    result = jnp.array(sorted(_map_nonzero_recall_to_all_positions(1, presentation).tolist()))
    assert jnp.all(result== jnp.array([0, 0, 0, 0, 1, 5]))

    tougher_result = jnp.array(sorted(map_recall_to_all_positions(1, presentation).tolist()))
    assert jnp.all(tougher_result == result)


def test_map_trial_to_positions():
    presentations = jnp.array([1, 2, 2, 3, 1, 3])
    trials = jnp.array([1, 3, 0,0,0,0])
    result = map_single_trial_to_positions(trials, presentations)

    result0 = jnp.array(sorted(result[0].tolist()))
    result1 = jnp.array(sorted(result[1].tolist()))
    result2 = jnp.array(sorted(result[2].tolist()))

    assert jnp.all(result0 == jnp.array([0, 0, 0, 0, 1, 5])) # item at study pos 1 also at pos 5
    assert jnp.all(result1 == jnp.array([0, 0, 0, 0, 2, 3])) # item at pos 3 also at pos 2
    assert jnp.all(result2 == jnp.array([0, 0, 0, 0, 0, 0])) # 0 is a non-recall, has no study pos
    

def test_numba_multitrial_item_repetitions():
    presentations = np.array([[1, 2, 2, 3, 1, 3], [1, 2, 3, 4, 5, 6]])
    trials = np.array([[1, 3, 0,0,0,0], [3, 1, 5,2,0,0]])
    result = multi_pres_spc(trials, presentations, presentations.shape[1])
    assert np.all(result == np.array([1., .5, .5, .5, 1., .5]))


def test_empty_list_length():
    """Tests the behavior when `list_length` is set to zero."""
    trials = jnp.array([[1, 2], [3, 4]])
    list_length = 0
    result = spc(trials, list_length)
    assert len(result) == 0


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("HealyKahana2014", "data['listtype'] == -1"),
        ("LohnasKahana2014", "data['list_type'] == 1"),
    ],
)
def test_single_pres_spc(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    numba_result = single_pres_spc(trials, list_length)
    jax_result = spc(jnp.array(trials), list_length)

    assert np.allclose(numba_result, jax_result)

    alternative_jax_result = spc(jnp.array(trials), jnp.array(presentations))
    assert np.allclose(numba_result, alternative_jax_result)

    alternative_jax_result = spc(jnp.array(trials), jnp.array(presentations[0]))
    assert np.allclose(numba_result, alternative_jax_result)


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("LohnasKahana2014", "data['list_type'] == 2"),
        ("LohnasKahana2014", "data['list_type'] == 3"),
        ("LohnasKahana2014", "data['list_type'] == 4"),
    ],
)
def test_multi_pres_spc(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    numba_result = multi_pres_spc(trials, presentations, list_length)
    jax_result = spc(jnp.array(trials), jnp.array(presentations))
    assert np.allclose(numba_result, jax_result)


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("HealyKahana2014", "data['listtype'] == -1"),
        ("LohnasKahana2014", "data['list_type'] == 1"),
    ],
)
def test_single_pres_pfr(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    numba_result = single_pres_pfr(trials, list_length)
    jax_result = pfr(jnp.array(trials), list_length)

    assert np.allclose(numba_result, jax_result)

    alternative_jax_result = pfr(jnp.array(trials), jnp.array(presentations))
    assert np.allclose(numba_result, alternative_jax_result)

    alternative_jax_result = pfr(jnp.array(trials), jnp.array(presentations[0]))
    assert np.allclose(numba_result, alternative_jax_result)


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("LohnasKahana2014", "data['list_type'] == 2"),
        ("LohnasKahana2014", "data['list_type'] == 3"),
        ("LohnasKahana2014", "data['list_type'] == 4"),
    ],
)
def test_multi_pres_pfr(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    numba_result = multi_pres_pfr(trials, presentations, list_length)
    jax_result = pfr(jnp.array(trials), jnp.array(presentations))
    assert np.allclose(numba_result, jax_result)
