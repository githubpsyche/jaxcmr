from jax import vmap, lax, numpy as jnp
from jaxcmr.helpers import ScalarInteger, Integer, Array

__all__ = [
    "item_occurrences",
    "_map_nonzero_recall_to_all_positions",
    "map_recall_to_all_positions",
    "map_single_trial_to_positions",
]

def item_occurrences(
    study_position_or_trial,
    presentation: Integer[Array, "list_length"],
):
    "For each item in study_position_or_trial, identifies positions of that item in presentation."
    return jnp.isin(presentation, study_position_or_trial).astype(float)

def find_item_positions(item_index, presentation):
    """Finds positions of target_value in the presentation array."""
    return jnp.where(
        presentation == item_index, jnp.arange(1, presentation.shape[0] + 1), 0
    )


def _map_nonzero_recall_to_all_positions(recall_pos, presentation):
    """Maps a single recall position to all possible presentation positions if recall_pos > 0."""
    target_item = presentation[recall_pos - 1]
    return find_item_positions(target_item, presentation)


def map_recall_to_all_positions(recall_pos, presentation):
    return lax.cond(
        recall_pos == 0,
        lambda _: jnp.zeros(presentation.shape[0], dtype=jnp.int32),
        lambda _: _map_nonzero_recall_to_all_positions(recall_pos, presentation),
        None,
    )


def map_single_trial_to_positions(trial, presentation):
    """Maps all recall positions in a single trial to their presentation positions."""
    return vmap(map_recall_to_all_positions, in_axes=(0, None))(trial, presentation)
