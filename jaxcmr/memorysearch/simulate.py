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
    study_events,
    recall_events,
    trial_count
)
from jax import jit, random, lax

# %% Exports

__all__ = ["simulate_trial", "free_recall", "simulate_trials"]

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
    model_create_fn: Callable,
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
    model_create_fn: Callable,
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
    model_create_fn: Callable,
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


@partial(jit, static_argnums=(0, 1, 2))
@dispatch
def simulate_trials(
    model_create_fn: Callable,
    item_count: ScalarInteger,
    experiment_count: ScalarInteger,
    presentation: Integer[Array, "study_events"],
    rng: PRNGKeyArray,
    parameters: dict,
) -> Integer[Array, "trial_count recall_events"]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_create_fn(item_count, presentation.shape[0], parameters)
    model = start_retrieving(experience(model, presentation))
    return lax.map(
        lambda split_rng: free_recall(model, split_rng)[1],
        random.split(rng, experiment_count),
    )