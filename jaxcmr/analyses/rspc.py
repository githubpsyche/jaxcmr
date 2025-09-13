"""Relative serial position curve utilities.

Compute a serial position curve where each recall is scored “correct” 
if it is exactly one position after the previous recall (previous + 
1), with the first recall scored relative to position 0 (so only a 
recall of study position 1 is correct).
"""

__all__ = ["Tabulation", "tabulate_trial", "relative_spc", "plot_relative_spc"]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, lax, vmap
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import init_plot, plot_data, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


class Tabulation(Pytree):
    """Track whether recalls follow their immediate study neighbor."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )
        self.previous = self.item_study_positions[first_recall - 1]
        self.recalled = (
            jnp.zeros(self.list_length, dtype=bool)
            .at[0]
            .set(jnp.any(self.previous == 1))
        )

    def tabulate_transition(self, recall: Int_) -> Bool[Array, " positions"]:
        """Returns position flags when recall follows a +1 neighbor.

        Args:
            recall: Recalled item number (1-based).
        """
        recall_positions = self.item_study_positions[recall - 1]
        prev_plus_one = jnp.array(jnp.where(self.previous > 0, self.previous + 1, 0))

        matches = (
            (recall_positions[:, None] == prev_plus_one[None, :])
            & (recall_positions[:, None] > 0)
            & (prev_plus_one[None, :] > 0)
        )

        matched_rows = jnp.any(matches, axis=1)
        positions_to_flag = jnp.array(jnp.where(matched_rows, recall_positions, 0))
        idx = jnp.clip(positions_to_flag - 1, 0, self.list_length - 1)
        return jnp.zeros_like(self.recalled).at[idx].set(positions_to_flag > 0)

    def tabulate(self, recall: Int_) -> "Tabulation":
        """Returns tabulation updated with a single recall.

        Args:
            recall: Recalled item number (1-based).
        """
        return self.replace(
            previous=self.item_study_positions[recall - 1],
            recalled=self.recalled | self.tabulate_transition(recall),
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> Float[Array, " study_events"]:
    """Returns study-position flags for recalls following their serial predecessor.

    Args:
        trial: Recalled item numbers in retrieval order; 0 pads unused events.
        presentation: Presented item numbers in study order.
        size: Maximum number of study positions an item may occupy.
    """
    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.recalled


def relative_spc(
    recalls: Integer[Array, " trial_count recall_positions"],
    presentations: Integer[Array, " trial_count study_positions"],
    list_length: Optional[int] = None,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Returns relative serial position accuracy by study position.

    Args:
        recalls: Trial-by-position recalled item numbers. 0 pads unused events.
        presentations: Trial-by-position presented item numbers.
        list_length: Length of the study list.
        size: Maximum number of study positions an item may occupy.
    """
    scores = vmap(tabulate_trial, in_axes=(0, 0, None))(recalls, presentations, size)
    return jnp.mean(scores, axis=0)


def plot_relative_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    distances: Optional[Float[Array, " word_count word_count"]] = None,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
) -> Axes:
    """Returns axis with plotted relative serial position curve.

    Args:
        datasets: Trial data for plotting.
        trial_masks: Masks to filter trials in each dataset.
        distances: Unused; kept for signature compatibility.
        color_cycle: Colors for each dataset.
        labels: Legend labels for datasets.
        contrast_name: Name of contrast for labeling.
        axis: Pre-existing axis to plot on.
        size: Maximum number of study positions an item may occupy.
    """
    axis = init_plot(axis)

    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]

    if labels is None:
        labels = [""] * len(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    if isinstance(trial_masks, jnp.ndarray):
        trial_masks = [trial_masks]

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(relative_spc, static_argnames=("size")),
                size,
            )
        )

        color = color_cycle.pop(0)
        plot_data(
            axis,
            jnp.arange(max_list_length, dtype=int) + 1,
            subject_values,
            labels[data_index],
            color,
        )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
