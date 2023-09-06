# %% Imports

from jaxcmr.helpers import (
    Integer,
    Float,
    Array,
    ScalarInteger,
    ScalarFloat,
    study_events,
    recall_events,
    get_item_count,
    log_likelihood,
)
from beartype.typing import Tuple, Callable
from jax import jit, lax, vmap, numpy as jnp
from plum import dispatch
from functools import partial
from jaxcmr.memorysearch import (
    MemorySearch,
    retrieve,
    start_retrieving,
    experience,
    outcome_probability,
)
import numpy as np
from jax.tree_util import Partial

# %% Public interface

__all__ = [
    "predict_and_simulate_retrieval",
    "predict_trial",
    "init_and_predict_trial",
    "predict_trials",
    "create_predict_fn",
]

# %% Single Event Prediction (and Simulation)


@jit
@dispatch
def predict_and_simulate_retrieval(
    model: MemorySearch, choice: ScalarInteger
) -> Tuple[MemorySearch, ScalarFloat]:
    """Predict and simulate a specified retrieval outcome given a model state"""
    return retrieve(model, choice), outcome_probability(model, choice)


# %% Trial Prediction - Predict Each Event in a Trial


@jit
@dispatch
def predict_trial(
    model: MemorySearch, trial: Integer[Array, "recall_events"]
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Predict a chain of retrieval outcomes (trial) given a model state"""
    return lax.scan(predict_and_simulate_retrieval, model, trial)


@partial(jit, static_argnums=(0, 1))
@dispatch
def init_and_predict_trial(
    model_init: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    parameters: dict,
    trial: Integer[Array, "recall_events"],
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Predict a recall sequence given parameters for a model"""
    model = start_retrieving(experience(model_init(item_count, item_count, parameters)))
    return lax.scan(predict_and_simulate_retrieval, model, trial)


@partial(jit, static_argnums=(0, 1))
@dispatch
def init_and_predict_trial(
    model_init: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    presentation: Integer[Array, "study_events"],
    trial: Integer[Array, "recall_events"],
    parameters: dict,
) -> Tuple[MemorySearch, Float[Array, "recall_events"]]:
    """Predict a recall sequence (trial) given model parameters and a presentation sequence"""
    model = model_init(item_count, presentation.shape[0], parameters)
    model = start_retrieving(experience(model, presentation))
    return lax.scan(predict_and_simulate_retrieval, model, trial)


# %% Multi-trial Prediction


@partial(jit, static_argnums=(0, 1))
@dispatch
def predict_trials(
    model_init: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    trials: Integer[Array, "trial_count event_count"],
    parameters: dict,
) -> ScalarFloat:
    """Log-likelihood of a set of independent recall sequences given parameters for a model"""
    model = start_retrieving(experience(model_init(item_count, item_count, parameters)))
    return log_likelihood(vmap(predict_trial, in_axes=(None, 0))(model, trials)[1])


@partial(jit, static_argnums=(0, 1))
@dispatch
def predict_trials(
    model_init: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    item_count: ScalarInteger,
    presentations: Integer[Array, "trial_count study_events"],
    trials: Integer[Array, "trial_count event_count"],
    parameters: dict,
) -> ScalarFloat:
    """Log-likelihood of trials given matched presentation sequences and model parameters"""
    return log_likelihood(
        vmap(init_and_predict_trial, in_axes=(None, None, 0, 0, None))(
            model_init, item_count, presentations, trials, parameters
        )[1]
    )


# %% Factory Functions

def create_predict_fn(
    model_init: Callable[[ScalarInteger, ScalarInteger, dict], MemorySearch],
    presentations: Integer[Array, "trial_count study_events"],
    trials: Integer[Array, "trial_count recall_event_count"],
    ) -> Callable[[dict], ScalarFloat]:
    """
    Configure a fn returning the log-likelihood of a set of trials given a set of parameters.

    When presentations are identical across trials, predict_trials can be partially applied to 
    model_init and trials plus the appropriate item count.

    When presentations vary across trials but item counts are identical, predict_trials can be
    partially applied to model_init and trials plus the appropriate item count.

    When presentations and item counts vary across trials, a list of partially configured
    predict_trials functions can be created with a list comprehension; one for each item count.
    """

    item_counts = vmap(get_item_count)(presentations)
    unique_item_counts = np.unique(item_counts)
    item_counts_are_identical = len(unique_item_counts) == 1
    presentations_are_identical = len(np.unique(presentations).shape) == 1

    if item_counts_are_identical and presentations_are_identical:
        return Partial(predict_trials, model_init, item_counts[0].item(), trials)
    elif item_counts_are_identical:
        return Partial(predict_trials, model_init, item_counts[0].item(), presentations, trials)
    else:
        functions = [
            Partial(predict_trials, model_init, item_count, presentations[item_counts==item_count], trials[item_counts==item_count])
            for item_count in unique_item_counts
        ]

        @jit
        def predict_trials_fn(parameters: dict) -> ScalarFloat:
            return sum(fn(parameters) for fn in functions)
        
        return predict_trials_fn
