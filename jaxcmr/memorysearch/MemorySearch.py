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

#%% Imports

from jaxtyping import Integer, Float, Array, PRNGKey, Bool
from typing import Tuple
from flax.struct import PyTreeNode
from plum import dispatch
from jax import jit, random, lax, numpy as jnp

#%% Public interface

__all__ = [
    'MemorySearch',
    'item_count',
    'experience',
    'outcome_probabilities',
    'outcome_probability',
    'start_retrieving',
    'active',
    'stop_recall',
    'retrieve_item',
    'retrieve',
    'single_free_recall',
    'maybe_single_free_recall',
    'free_recall',
    'predict_and_simulate_retrieval',
    'predict_and_simulate_trial',
    'uniform_presentations_data_likelihood',
    'variable_presentations_data_likelihood'
    ]

#%% Types

class MemorySearch(PyTreeNode):
    pass

#%% Accessors

@dispatch.abstract
def get_item_count(model: MemorySearch) -> int | Integer[Array, ""]:
    "Return the number of items initialized with the model"

#%% Encoding

@dispatch.abstract
def experience(model: MemorySearch, choice: int | Integer[Array, ""]) -> MemorySearch:
    "Experience a study item at the specified index (1-indexed) or ignore it (choice = 0)"

@dispatch
def experience(model: MemorySearch):
    "Experience all study items initialized with the model"
    return lax.fori_loop(1, get_item_count(model)+1, lambda i, model: experience(model, i), model)

@dispatch
def experience(model: MemorySearch, choices: Integer[Array, "presentation_count"]):
    "Experience all study items initialized with the model, in the specified order"
    return lax.fori_loop(0, choices.shape[0], lambda i, model: experience(model, choices[i]), model)

#%% Event Probabilities

@dispatch.abstract
def outcome_probabilities(model: MemorySearch) -> Float[Array, "item_count+1"]:
    "Return the probability of each possible retrieval outcome"

@dispatch.abstract
def outcome_probability(
    model: MemorySearch, choice: int | Integer[Array, ""]) -> float | Float[Array, ""]:
    "Return the probability of a particular retrieval outcome"

#%% Item Retrieval

@dispatch.abstract
def start_retrieving(model: MemorySearch) -> MemorySearch:
    "Shift to retrieval mode"

@dispatch.abstract
def is_active(model: MemorySearch) -> bool | Bool[Array, ""]:
    "Return whether the model has finished retrieving"

@dispatch.abstract
def stop_recall(model: MemorySearch, _: int | Integer[Array, ""]) -> MemorySearch:
    "Stop recalling"

@dispatch.abstract
def retrieve_item(model: MemorySearch, choice: int | Integer[Array, ""]) -> MemorySearch:
    "Retrieve an item"

@jit
@dispatch
def retrieve(model: MemorySearch, choice: int | Integer[Array, ""]) -> MemorySearch:
    "Perform specified retrieval event, either item recall (choice > 0) or termination (choice = 0)"
    return lax.cond(choice > 0, retrieve_item, stop_recall, model, choice)

#%% Free Recall

@jit
@dispatch
def single_free_recall(model: MemorySearch, rng: PRNGKey) -> Tuple[MemorySearch, int]:
    "Perform a free recall event and return the resulting state."
    outcome_probabilities = outcome_probabilities(model)
    choice = random.choice(rng, outcome_probabilities.shape[0], p=outcome_probabilities)
    return retrieve(model, choice), choice

@jit
@dispatch
def maybe_single_free_recall(model: MemorySearch, rng: PRNGKey) -> Tuple[MemorySearch, int]:
    "Perform a free recall event if the model is active and return the resulting state."
    return lax.cond(is_active(model), single_free_recall, lambda model, _: (model, 0), (model, rng))

@jit
@dispatch
def free_recall(model: MemorySearch, rng: PRNGKey) -> Tuple[MemorySearch, int]:
    "Perform free recall events until the model is inactive and return the resulting state."
    return lax.scan(maybe_single_free_recall, model, random.split(rng, get_item_count(model)))

#%% Trial Probabilities

lb = jnp.finfo(float).eps

@jit
@dispatch
def predict_and_simulate_retrieval(
    model: MemorySearch, 
    choice: int | Integer[Array, ""]
) -> Tuple[MemorySearch, float | Float[Array, ""]]:
    "Predict the probability of a particular retrieval outcome and then simulate that outcome"
    likelihood = lax.cond(
        is_active(model), outcome_probability, lambda model, _: 0.0, (model, choice))
    return retrieve(model, choice), likelihood + lb

@jit
@dispatch
def predict_and_simulate_trial(
    model: MemorySearch,
    trial: Float[Array, "event_count"]
) -> Tuple[MemorySearch, Float[Array, "event_count"]]:
    "Predict the probability of each retrieval outcome and then simulate the outcome of each event"
    return lax.scan(predict_and_simulate_retrieval, model, trial)

@jit
@dispatch
def uniform_presentations_data_likelihood(
    model: MemorySearch,
    trials: Integer[Array, "trial_count event_count"],
    ) -> float | Float[Array, ""]:
    "Log-likelihood over trials with uniform presentation structure for an initialized model"
    model = start_retrieving(experience(model))
    return -jnp.sum(jnp.log(lax.map(
        lambda trial: predict_and_simulate_trial(model, trial)[1], trials
        )))

@jit
@dispatch
def variable_presentations_data_likelihood(
    model: MemorySearch,
    presentations: Integer[Array, "trial_count max_presentation_count"],
    trials: Integer[Array, "trial_count event_count"],
    ) -> float | Float[Array, ""]:
    "Log-likelihood over trials with variable presentation structure for an initialized model"
    models = lax.map(f=lambda presentation: experience(model, presentation), xs=presentations)
    models = lax.map(f=start_retrieving, xs=models)
    return -jnp.sum(jnp.log(lax.map(
        f=lambda i: predict_and_simulate_trial(models[i], trials[i]), xs=jnp.arange(trials.shape[0])
        )))