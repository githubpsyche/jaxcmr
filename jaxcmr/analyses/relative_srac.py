"""Relative Serial Recall Accuracy Curve.

Computes the probability of recalling the correct item at each
output position relative to the expected serial position, accounting
for items that may repeat across study positions.

"""

__all__ = ["Tabulation", "tabulate_trial", "relative_srac", "plot_relative_srac"]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, lax, vmap
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
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
        self.previous = lax.cond(
            first_recall > 0,
            lambda: self.item_study_positions[first_recall - 1],
            lambda: jnp.zeros((self.size,), dtype=int),
        )
        self.recalled = (
            jnp.zeros(self.list_length, dtype=bool)
            .at[0]
            .set(jnp.any(self.previous == 1))
        )

    def tabulate_transition(self, recall: Int_) -> Bool[Array, " positions"]:
        """Position flags when recall follows a +1 neighbor."""
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
        """Update tabulation with a single recall."""
        return self.replace(
            previous=self.item_study_positions[recall - 1],
            recalled=self.recalled | self.tabulate_transition(recall),
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> Float[Array, " study_events"]:
    """Flag study positions where recall follows its predecessor.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        1-indexed recalls; 0 pads unused events.
    presentation : Integer[Array, " study_events"]
        Item IDs in study order.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_events"]
        True at positions recalled after their neighbor.

    """
    trial = trial[:presentation.size]
    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.recalled


def relative_srac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Relative serial recall accuracy by study position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean accuracy at each study position.

    """
    scores = vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"],
        dataset["pres_itemnos"],
        size,
    )
    return jnp.mean(scores, axis=0)


def plot_relative_srac(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,

) -> Axes:
    """Plot relative serial recall accuracy with intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the relative SRAC plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(datasets, trial_masks, color_cycle, axis)


    if labels is None:
        labels = [""] * len(datasets)



    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(relative_srac, static_argnames=("size")),
                size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            jnp.arange(max_list_length, dtype=int) + 1,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
