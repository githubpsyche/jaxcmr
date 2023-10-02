"""
"""

# %% Imports

from jaxcmr._analyses import _map_nonzero_recall_to_all_positions, map_recall_to_all_positions, map_single_trial_to_positions, item_occurrences, crp
from jaxcmr.datasets import load_data, generate_trial_mask
from jaxcmr.analyses import (
    single_pres_crp,
    multi_pres_crp,
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


# %% 


# %% Integration Tests

@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("HealyKahana2014", "data['listtype'] == -1"),
        ("LohnasKahana2014", "data['list_type'] == 1"),
    ],
)
def test_numba_crp(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    single_pres_result = single_pres_crp(trials, list_length)
    multi_pres_result = multi_pres_crp(trials, presentations, list_length)

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

    single_pres_result = single_pres_crp(trials, list_length)
    multi_pres_result = multi_pres_crp(trials, presentations, list_length)

    assert not np.allclose(single_pres_result, multi_pres_result)


@pytest.mark.parametrize(
    "data_tag, trial_query",
    [
        ("HealyKahana2014", "data['listtype'] == -1"),
        ("LohnasKahana2014", "data['list_type'] == 1"),
    ],
)
def test_single_pres_crp(data_tag, trial_query):
    trials = recalls(data_tag, trial_query)
    presentations = pres_itemnos(data_tag, trial_query)
    list_length = presentations.shape[1]

    numba_result = single_pres_crp(trials, list_length)
    jax_result = crp(jnp.array(trials), list_length)

    assert np.allclose(numba_result, jax_result)

    alternative_jax_result = crp(jnp.array(trials), jnp.array(presentations))
    assert np.allclose(numba_result, alternative_jax_result)

    alternative_jax_result = crp(jnp.array(trials), jnp.array(presentations[0]))
    assert np.allclose(numba_result, alternative_jax_result)