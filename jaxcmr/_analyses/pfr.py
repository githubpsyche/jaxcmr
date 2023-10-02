"""
Probability of first recall (PFR)

The probability of recalling an item first in a trial as a function of its serial position in the study list. Especially in delayed free recall, The PFR typically favors the first few items in the list, and then drops off sharply. In immediate free recall, the PFR can (also) favor the last few items in the list due to recency effects. The PFR is calculated as the proportion of trials in which an item was recalled first, across all trials in which the item was recalled at all.
"""

# %% Imports
from jax import numpy as jnp, vmap, lax
from plum import dispatch
from jaxcmr._analyses.map_to_study_positions import map_recall_to_all_positions, item_occurrences
from jaxcmr.helpers import ScalarInteger, Integer, Array, recall_events, study_events, trial_count

# %% Exports
__all__ = ["pfr"]

# %% PFR Function

@dispatch
def pfr(
    trials: Integer[Array, "trial_count recall_events"],
    list_length: ScalarInteger,
):
    presentation = jnp.arange(1, list_length + 1)
    return jnp.mean(vmap(item_occurrences, in_axes=(0, None))(trials[:, 0], presentation), axis=0)


@dispatch   
def pfr(
    trials: Integer[Array, "trial_count recall_events"],
    presentations: Integer[Array, "trial_count study_events"],
):
    """
    Calculate the Probability of First Recall (PFR) across multiple trials.
    
    trials: Array of shape [num_trials, num_recalls] 
            1-indexed study positions in the order items were recalled
    presentations: Array of shape [num_trials, list_length] 
            1-indexed item indices corresponding to order in the study sequence
    """
    
    map_fn = vmap(map_recall_to_all_positions, in_axes=(0, 0))
    mapped_trials = map_fn(trials[:, 0], presentations) # Shape: [num_trials, list_length]
    return jnp.sum(mapped_trials > 0, axis=0)/trials.shape[0] 


@dispatch   
def pfr(
    trials: Integer[Array, "trial_count recall_events"],
    presentation: Integer[Array, "study_events"],
):
    """
    Calculate the Probability of First Recall (PFR) across multiple trials.
    
    trials: Array of shape [num_trials, num_recalls] 
            1-indexed study positions in the order items were recalled
    presentations: Array of shape [num_trials, list_length] 
            1-indexed item indices corresponding to order in the study sequence
    """
    
    map_fn = vmap(map_recall_to_all_positions, in_axes=(0, None))
    mapped_trials = map_fn(trials[:, 0], presentation) # Shape: [num_trials, list_length]
    return jnp.sum(mapped_trials > 0, axis=0)/trials.shape[0] 