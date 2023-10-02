"""
The serial position curve (SPC)

Recall rate as a function of serial position is a fundamental property of human memory. The serial position curve (SPC) is a function that describes the relationship between the serial position of an item in a list and the probability that the item will be recalled. The SPC is typically U-shaped, with the highest recall rates for items at the beginning and end of the list, and the lowest recall rates for items in the middle of the list. The SPC is a fundamental property of human memory, and is often used as a benchmark for evaluating models of memory.
"""

# %% Imports
from jax import numpy as jnp, vmap, lax
from plum import dispatch
from jaxcmr._analyses.map_to_study_positions import map_single_trial_to_positions, item_occurrences
from jaxcmr.helpers import ScalarInteger, Integer, Array, recall_events, study_events, trial_count

# %% Exports
__all__ = ["spc"]


# %% SPC Function

@dispatch
def spc(
    trials: Integer[Array, "trial_count recall_events"],
    list_length: ScalarInteger,
):
    presentation = jnp.arange(1, list_length + 1)
    return jnp.mean(vmap(item_occurrences, in_axes=(0, None))(trials, presentation), axis=0)


@dispatch
def spc(
    trials: Integer[Array, "trial_count recall_events"],
    presentations: Integer[Array, "trial_count study_events"],
):
    """
    Calculate the Serial Position Curve (SPC) across multiple trials.
    
    trials: Array of shape [num_trials, num_recalls] 
            1-indexed study positions in the order items were recalled
    presentations: Array of shape [num_trials, list_length] 
            1-indexed item indices corresponding to order in the study sequence
    """
    
    # Step 1: Map recall positions to study positions for each trial
    map_fn = vmap(map_single_trial_to_positions, in_axes=(0, 0))
    mapped_trials = map_fn(trials, presentations)  # Shape: [num_trials, num_recalls, list_length]
    
    # Step 2: Count recalls for each position
    counts = jnp.sum(jnp.any(mapped_trials > 0, axis=1), axis=0)  # Shape: [list_length]
    
    # Step 3: Normalize to get recall probability
    num_trials = trials.shape[0]
    return counts / num_trials


@dispatch
def spc(
    trials: Integer[Array, "trial_count recall_events"],
    presentation: Integer[Array, "study_events"],
):
    """
    Calculate the Serial Position Curve (SPC) across multiple trials with the same study sequence.
    
    trials: Array of shape [num_trials, num_recalls] 
            1-indexed study positions in the order items were recalled
    presentations: Array of shape [list_length] 
            1-indexed item indices corresponding to order in the study sequence
    """
    
    # Step 1: Map recall positions to study positions for each trial
    map_fn = vmap(map_single_trial_to_positions, in_axes=(0, None))
    mapped_trials = map_fn(trials, presentation)  # Shape: [num_trials, num_recalls, list_length]
    
    # Step 2: Count recalls for each position
    counts = jnp.sum(jnp.any(mapped_trials > 0, axis=1), axis=0)  # Shape: [list_length]
    
    # Step 3: Normalize to get recall probability
    num_trials = trials.shape[0]
    return counts / num_trials
