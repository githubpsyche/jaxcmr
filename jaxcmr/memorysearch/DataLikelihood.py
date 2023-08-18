# %% Imports

from jaxtyping import Integer, Float, Array, PRNGKeyArray
from typing import Tuple
from jax import jit, random, lax, numpy as jnp
from plum import dispatch
from jaxcmr.memorysearch.MemorySearch import *

# %% Public interface

__all__ = [
    'predict_and_simulate_retrieval',
    'predict_and_simulate_trial',
    'uniform_presentations_data_likelihood',
    'variable_presentations_data_likelihood',
]

# %% Functions

@jit
@dispatch
def predict_and_simulate_retrieval(
    model: MemorySearch, choice: int | Integer[Array, ""]
) -> Tuple[MemorySearch, float | Float[Array, ""]]:
    "Predict the probability of a particular retrieval outcome and then simulate that outcome"
    return retrieve(model, choice), outcome_probability(model, choice)


@jit
@dispatch
def predict_and_simulate_trial(
    model: MemorySearch, 
    trial: Integer[Array, "event_count"]
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
    return -jnp.sum(
        jnp.log(
            lax.map(lambda trial: predict_and_simulate_trial(model, trial)[1], trials)
        )
    )


@jit
@dispatch
def variable_presentations_data_likelihood(
    model: MemorySearch,
    presentations: Integer[Array, "trial_count max_presentation_count"],
    trials: Integer[Array, "trial_count event_count"],
) -> float | Float[Array, ""]:
    "Log-likelihood over trials with variable presentation structure for an initialized model"
    models = lax.map(
        f=lambda presentation: experience(model, presentation), xs=presentations
    )
    models = lax.map(f=start_retrieving, xs=models)
    return -jnp.sum(
        jnp.log(
            lax.map(
                f=lambda i: predict_and_simulate_trial(models[i], trials[i]),
                xs=jnp.arange(trials.shape[0]),
            )
        )
    )