"""
Trial simulation functions for memory search models.

Memory search involves encoding items into memory and eventually performing a sequence of retrieval operations to recall items from memory.

Outcome probability functions provide a probability distribution over possible retrieval outcomes given a state of the model.

Here we provide functions for simulating sequences of retrieval events in a model by sampling from the outcome probability distribution at each step and updating the model state accordingly.
"""

# %% Imports
from functools import partial
from plum import dispatch
from jaxcmr.memorysearch.types import MemorySearch
from jaxcmr.memorysearch.outcome_probability import outcome_probability
from jaxcmr.memorysearch.experience import experience
from jaxcmr.memorysearch.retrieve import retrieve, start_retrieving
from beartype.typing import Tuple, Callable
from jaxcmr.helpers import (
    PRNGKeyArray,
    ScalarInteger,
    Integer,
    Array,
    Bool,
    study_events,
    recall_events,
    trial_count,
    get_item_count,
    select_parameters_by_subject,
    tree_transpose,
    recall_by_study_position
)
import numpy as  np
import numpy.matlib
from jax import jit, random, lax, vmap, numpy as jnp

# %% Exports

__all__ = ["free_recall", "simulate_trial", "simulate_trials", "simulate_h5_from_h5"]

# %% Single event simulation


@jit
@dispatch
def single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger]:
    """Perform a free recall event and return the resulting state."""
    p_all = outcome_probability(model)
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return retrieve(model, choice), choice


@jit
@dispatch
def maybe_single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger]:
    """Perform a free recall event if the model is active and return the resulting state."""
    return lax.cond(
        model.is_active, single_free_recall, lambda m, _: (m, 0), model, rng
    )


@jit
@dispatch
def free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger | PRNGKeyArray]:
    """Perform free recall events until the model is inactive and return the resulting state."""
    return lax.scan(
        maybe_single_free_recall, model, random.split(rng, model.item_count)
    )


# %% Trial simulation


@partial(jit, static_argnums=(0, 1))
@dispatch
def simulate_trial(
    model_create_fn: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    presentation: Integer[Array, "study_events"],
    rng: PRNGKeyArray,
    parameters: dict,
) -> Integer[Array, "recall_events"]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_create_fn(item_count, presentation.shape[0], parameters)
    model = start_retrieving(experience(model, presentation))
    return free_recall(model, rng)[1]


@partial(jit, static_argnums=(0, 1))
@dispatch
def simulate_trial(
    model_create_fn: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    rng: PRNGKeyArray,
    parameters: dict,
) -> Integer[Array, "recall_events"]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_create_fn(item_count, item_count, parameters)
    model = start_retrieving(experience(model))
    return free_recall(model, rng)[1]


@partial(jit, static_argnums=(0, 1, 2))
@dispatch
def simulate_trials(
    model_create_fn: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    experiment_count: ScalarInteger,
    rng: PRNGKeyArray,
    parameters: dict,
) -> Integer[Array, "trial_count recall_events"]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_create_fn(item_count, item_count, parameters)
    model = start_retrieving(experience(model))
    return lax.map(
        lambda split_rng: free_recall(model, split_rng)[1],
        random.split(rng, experiment_count),
    )


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
    model_create_fn: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
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
    parameters_by_index = select_parameters_by_subject(subject_indices, parameters)
    item_counts = vmap(get_item_count)(presentations)

    result = vmap(simulate_trial, in_axes=(None, None, 0, 0, 0))(
        model_create_fn,
        max(item_counts).item(),
        presentations,
        random.split(rng, len(presentations)),
        tree_transpose(parameters_by_index),
    )
    sim_h5["recalls"][:, :result.shape[1]] = vmap(recall_by_study_position)(presentations, result)

    return sim_h5
