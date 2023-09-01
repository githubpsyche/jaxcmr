# %% Imports
from jaxcmr.helpers import (
    tree_transpose,
    PRNGKeyArray,
    ScalarInteger,
    Integer,
    Array,
    trial_count,
    Bool,
    get_item_count,
    recall_by_study_position,
    select_parameters_by_subject,
)
import numpy.matlib
import numpy as np
from beartype.typing import Callable
from jax import vmap, random, numpy as jnp
from jaxcmr.memorysearch.simulate import simulate_trial

# %% Exports

__all__ = ["simulate_h5_from_h5"]

# %% Full dataset simulation


def preallocate_for_h5_dataset(
    data: dict, trial_mask: Bool[Array, "trial_count"], experiment_count: ScalarInteger
):
    """Pre-allocate a dictionary of arrays for a given trial mask."""
    return {
        key: np.matlib.repmat(
            np.zeros(np.shape(data["pres_itemnos"][trial_mask]), dtype=np.int64),
            experiment_count,
            1,
        )
        if key == "recalls"
        else np.matlib.repmat(data[key][trial_mask], experiment_count, 1)
        for key in data
    }


def simulate_h5_from_h5(
    model_create_fn: Callable,
    data: dict,
    parameters: list[dict],
    rng: PRNGKeyArray,
    trial_mask: Bool[Array, "trial_count"],
    experiment_count: ScalarInteger,
):
    """Simulate a dataset from a dataset and specified model constructor and parameters."""
    sim_h5 = preallocate_for_h5_dataset(data, trial_mask, experiment_count)
    presentations = np.matlib.repmat(
        data["pres_itemnos"][trial_mask], experiment_count, 1
    )
    subject_indices = np.matlib.repmat(data["subject"][trial_mask], experiment_count, 1)
    item_counts = vmap(get_item_count)(presentations)
    _simulate_trials = vmap(simulate_trial, in_axes=(None, None, 0, 0, 0))

    for item_count in jnp.unique(item_counts):
        rng, _ = random.split(rng)
        ic_mask = item_counts == item_count
        ic_presentations = presentations[ic_mask]
        ic_parameters = select_parameters_by_subject(
            subject_indices[ic_mask], parameters
        )

        result = _simulate_trials(
            model_create_fn,
            item_count.item(),
            ic_presentations,
            random.split(rng, len(ic_presentations)),
            tree_transpose(ic_parameters),
        )
        sim_h5["recalls"][ic_mask, : result.shape[1]] = vmap(recall_by_study_position)(
            ic_presentations, result
        )

    return sim_h5
