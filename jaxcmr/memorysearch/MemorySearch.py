"""
MemorySearch

Abstract type and functions for memory search models.
Subtypes that implement all abstract functions are compatible with concrete functions defined here.

A MemorySearch model is a stateful model that can be in one of two modes: acthive or inactive.
A model is initialized in association with a specific item_count, and can be updated by experiencing
additional items or by retrieving items from memory.
Items are specified for encoding and retrieval by their index (1-indexed).
0 is a special index that indicates based on context either that no item is encoded or that 
retrieval is terminated.
Given a state of the model, a probability of each possible retrieval outcome can be computed.
"""

# %% Imports

from jaxcmr.helpers import (
    replace, Integer, Float, Array, ScalarInteger, ScalarFloat,
    PRNGKeyArray, study_events, recall_outcomes, recall_events
)
from typing import Tuple
from simple_pytree import Pytree
from plum import dispatch
from jax import jit, random, lax, numpy as jnp
from functools import partial
from beartype.typing import Callable

lb = jnp.finfo(float).eps


# %% Public interface

__all__ = [
    "MemorySearch",
    "experience",
    "outcome_probabilities",
    "outcome_probability",
    "start_retrieving",
    "stop_recall",
    "retrieve_item",
    "retrieve",
    "single_free_recall",
    "maybe_single_free_recall",
    "free_recall",
    "exponential_primacy_weighting",
    "simulate_trial"
]


# %% Types


class MemorySearch(Pytree, mutable=True):
    item_count: int  # the number of items initialized with the model
    is_active: bool  # whether the model is still open to new experiences or retrieval events


@partial(jit, static_argnums=(0,))
@dispatch
def exponential_primacy_weighting(
        presentation_count: ScalarInteger,
        primacy_scale: ScalarFloat,
        primacy_decay: ScalarFloat,
) -> Float[Array, "study_events"]:
    """The primacy effect as exponential decay of boosted attention weights."""
    arange = jnp.arange(presentation_count, dtype=jnp.float32)
    return primacy_scale * jnp.exp(-primacy_decay * arange) + 1


# %% Encoding


@dispatch.abstract
def experience_item(model: MemorySearch, item_index: ScalarInteger) -> MemorySearch:
    """Experience a study item at the specified index"""


@dispatch
def experience(model: MemorySearch, choices: Integer[Array, "study_events"]):
    """Experience all study items initialized with the model, in the specified order"""
    return lax.fori_loop(
        0, choices.shape[0], lambda i, m: experience(m, choices[i]), model
    )


@jit
@dispatch
def experience(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Experience a study item at the specified index (1-indexed) or ignore it (choice = 0)"""
    return lax.cond(
        choice == 0, lambda _: model, lambda _: experience_item(model, choice - 1), None
    )


@dispatch
def experience(model: MemorySearch):
    """Experience all study items initialized with the model"""
    return lax.fori_loop(
        1, model.item_count + 1, lambda i, m: experience(m, i), model
    )


# %% Event Probabilities


@dispatch.abstract
def outcome_probabilities(model: MemorySearch) -> Float[Array, "recall_outcomes"]:
    """Return the probability of each possible retrieval outcome"""


@dispatch.abstract
def outcome_probability(model: MemorySearch, choice: ScalarInteger) -> ScalarFloat:
    """Return the probability of a particular retrieval outcome"""


# %% Item Retrieval


@dispatch.abstract
def start_retrieving(model: MemorySearch) -> MemorySearch:
    """Evolve model reflect its initial state at the start of free recall"""


@dispatch.abstract
def retrieve_item(
        model: MemorySearch, choice: ScalarInteger
) -> MemorySearch:
    """Retrieve an item from memory"""


@dispatch
def stop_recall(model: MemorySearch, _: ScalarInteger = 0) -> MemorySearch:
    """The model shifts to inactive mode"""
    return replace(model, is_active=False)


@jit
@dispatch
def retrieve(model: MemorySearch, choice: ScalarInteger) -> MemorySearch:
    """Perform specified retrieval event, either item recall (choice > 0) or termination (choice = 0)"""
    return lax.cond(choice > 0, retrieve_item, stop_recall, model, choice)


# %% Free Recall


@jit
@dispatch
def single_free_recall(
        model: MemorySearch,
        rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger]:
    """Perform a free recall event and return the resulting state."""
    p_all = outcome_probabilities(model)
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return retrieve(model, choice), choice


@jit
@dispatch
def maybe_single_free_recall(
        model: MemorySearch,
        rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger]:
    """Perform a free recall event if the model is active and return the resulting state."""
    return lax.cond(
        model.is_active, single_free_recall, lambda m, _: (m, 0), model, rng
    )


@jit
@dispatch
def free_recall(
        model: MemorySearch,
        rng: PRNGKeyArray
) -> Tuple[MemorySearch, ScalarInteger | PRNGKeyArray]:
    """Perform free recall events until the model is inactive and return the resulting state."""
    return lax.scan(
        maybe_single_free_recall, model, random.split(rng, model.item_count)
    )


# %% Data Simulation

@partial(jit, static_argnums=(0, 1))
@dispatch
def simulate_trial(
        model_create_fn: Callable,
        item_count: ScalarInteger,
        presentation: Integer[Array, "study_events"],
        rng: PRNGKeyArray,
        parameters: dict
) -> Integer[Array, "recall_events"]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_create_fn(item_count, presentation.shape[0], parameters)
    model = start_retrieving(experience(model, presentation))
    return free_recall(model, rng)[1]


# @partial(jit, static_argnums=(0, 1))
# @dispatch
# def simulate_trial(
#     model_create_fn: Callable,
#     item_count: ScalarInteger,
#     presentation: Integer[Array, "study_events"],
#     first_recall: ScalarInteger,
#     rng: PRNGKeyArray,
#     parameters: dict
# ) -> Float[Array, "recall_events"]:
#     """Initialize model and study events, then simulate and predict retrieval events"""
#     model = model_create_fn(item_count, presentation.shape[0], parameters)
#     model = start_retrieving(experience(model, presentation))
#     model = retrieve(model, first_recall)
#     return free_recall(model, rng)[1]
