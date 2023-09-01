# %% Imports

from jaxcmr.helpers import (
    Integer,
    Float,
    Array,
    ScalarInteger,
    ScalarFloat,
    study_events,
    recall_events,
    get_list_length,
    get_item_count,
    log_likelihood,
    recall_by_item_index
)
from beartype.typing import Tuple, Callable
from jax import jit, lax, numpy as jnp, vmap
from plum import dispatch
from functools import partial
from jaxcmr.memorysearch import MemorySearch, retrieve, start_retrieving, experience, outcome_probability
import numpy as np
from jax.tree_util import Partial

# %% Public interface

__all__ = [
    "predict_and_simulate_retrieval",
    "predict_and_simulate_trial",
    # "get_list_length",
    # "get_item_count",
    # "log_likelihood",
    # "recall_by_item_index",
    "predict_and_simulate_pres_and_trial",
    "uniform_presentations_data_likelihood",
    "variable_presentations_data_likelihood",
    
]


# %% Event-level likelihood functions


@jit
@dispatch
def predict_and_simulate_retrieval(
    model: MemorySearch, choice: ScalarInteger
) -> Tuple[MemorySearch, ScalarFloat]:
    """Predict the probability of a particular retrieval outcome and then simulate that outcome"""
    return retrieve(model, choice), outcome_probability(model, choice)


@jit
@dispatch
def predict_and_simulate_trial(
    model: MemorySearch, trial: Integer[Array, "recall_events"]
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Predict the probability of each retrieval outcome and then simulate the outcome of each event"""
    return lax.scan(predict_and_simulate_retrieval, model, trial)


@jit
@dispatch
def predict_and_simulate_trial(
    model_init: Callable,
    item_count: ScalarInteger,
    presentation: Integer[Array, "study_events"],
    trial: Integer[Array, "recall_events"],
    parameters: dict,
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    return predict_and_simulate_pres_and_trial(
        model_init, item_count, presentation, trial, parameters
    )


# %% Added flexibility for variable presentation structure


@partial(jit, static_argnums=(0, 1))
@dispatch
def predict_and_simulate_pres_and_trial(
    model_init,  #: Callable,
    item_count,  #: ScalarInteger,
    presentation,  #: Integer[Array, "study_events"],
    trial,  #: Integer[Array, "recall_events"],
    parameters,  #: dict,
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Initialize model and study events, then simulate and predict retrieval events"""
    model = model_init(item_count, presentation.shape[0], parameters)
    model = start_retrieving(experience(model, presentation))
    return predict_and_simulate_trial(model, trial)


# %% Multi-trial likelihood functions


@partial(jit, static_argnums=(0, 1))
@dispatch
def uniform_presentations_data_likelihood(
    model_create_fn: Callable,
    item_count: ScalarInteger,
    trials: Integer[Array, "trial_count event_count"],
    parameters,
) -> ScalarFloat:
    """Log-likelihood over trials with variable presentation structure for an uninitialized model"""
    model = model_create_fn(item_count, parameters)
    model = start_retrieving(experience(model))
    return log_likelihood(
        lax.map(lambda trial: predict_and_simulate_trial(model, trial)[1], trials)
    )


@partial(jit, static_argnums=(0, 1))
@dispatch
def variable_presentations_data_likelihood(
    model_create_fn: Callable,
    item_count: ScalarInteger,
    presentation: Integer[Array, "study_events"],
    trial: Integer[Array, "recall_events"],
    parameters,
) -> ScalarFloat:
    """Log-likelihood over trials with variable presentation structure for an uninitialized model"""

    return log_likelihood(
        predict_and_simulate_pres_and_trial(
            model_create_fn,
            item_count,
            presentation,
            trial,
            parameters,
        )[1]
    )


@partial(jit, static_argnums=(0, 1))
@dispatch
def variable_presentations_data_likelihood(
    model_create_fn: Callable,
    item_count: ScalarInteger,
    presentations: Integer[Array, "trial_count study_event_count"],
    trials: Integer[Array, "trial_count recall_event_count"],
    parameters,
) -> ScalarFloat:
    return log_likelihood(
        lax.map(
            lambda trial_index: predict_and_simulate_pres_and_trial(
                model_create_fn,
                item_count,
                presentations[trial_index],
                trials[trial_index],
                parameters,
            )[1],
            jnp.arange(trials.shape[0]),
        )
    )


def variable_presentations_likelihood(
    model_create_fn: Callable,
    item_count: ScalarInteger,
    presentations: Integer[Array, "trial_count study_event_count"],
    trials: Integer[Array, "trial_count recall_event_count"],
    parameters,
):
    return lax.map(
        lambda trial_index: predict_and_simulate_pres_and_trial(
            model_create_fn,
            item_count,
            presentations[trial_index],
            trials[trial_index],
            parameters,
        )[1],
        jnp.arange(trials.shape[0]),
    )


@dispatch
def variable_presentations_data_likelihood(
    model_create_fn: Callable,
    presentations: Integer[Array, "trial_count study_event_count"],
    trials: Integer[Array, "trial_count recall_event_count"],
) -> Callable:
    item_counts = vmap(get_item_count)(presentations)
    functions = [
        Partial(
            variable_presentations_likelihood,
            model_create_fn,
            item_count,
            presentations[item_counts == item_count],
            trials[item_counts == item_count],
        )
        for item_count in np.unique(item_counts)
    ]

    @jit
    def f(parameters) -> ScalarFloat:
        log_likelihoods = []
        for fn in functions:
            log_likelihoods.append(fn(parameters))
        return log_likelihood(jnp.vstack(log_likelihoods))

    return f
