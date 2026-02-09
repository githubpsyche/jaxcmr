"""Distance-binned CRP with conditional transition filtering.

Extends ``distcrp.DistanceTabulation`` with a per-event
``_should_tabulate`` mask so that only selected transitions
contribute to distance-bin counts, while all recalls still update
availability tracking.

"""

from __future__ import annotations

__all__ = [
    "DistanceTabulation",
    "tabulate_trial",
    "dist_crp",
    "plot_dist_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..math import cosine_similarity_matrix
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset
from .distcrp import compute_distance_bins_min_count


class DistanceTabulation(Pytree):
    """Per-bin transition counts with conditional tabulation."""

    def __init__(
        self,
        availability_mask: Bool[Array, " study_events"],
        first_recall: Int_,
        trial_distances: Float[Array, "study_events study_events"],
        bin_edges: Float[Array, " edges"],
    ):
        bin_count = bin_edges.shape[0] + 1
        self.bin_edges = bin_edges
        self.trial_distances = trial_distances
        self.actual_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_items = availability_mask
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall

    def _compute_transition_counts(
        self, current_item: Int_
    ) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
        """Compute actual and available bin counts for a transition.

        Parameters
        ----------
        current_item : Int_
            Currently recalled item (1-indexed).

        Returns
        -------
        tuple[Integer[Array, " bins"], Integer[Array, " bins"]]
            Updated actual and available bin counts.

        """
        distances_from_prev = self.trial_distances[self.previous_item - 1]
        actual_distance = distances_from_prev[current_item - 1]
        actual_bin = jnp.digitize(actual_distance, self.bin_edges)

        present_bins = jnp.digitize(distances_from_prev, self.bin_edges)
        bin_count = self.bin_edges.shape[0] + 1
        masked_bins = jnp.where(self.avail_items, present_bins, bin_count)
        bin_flags = jnp.zeros(bin_count + 1, dtype=jnp.int32).at[masked_bins].set(1)
        bin_flags = bin_flags[:-1]

        new_actual = self.actual_transitions.at[actual_bin].add(1)
        new_avail = self.avail_transitions + bin_flags
        return new_actual, new_avail

    def tabulate(self, choice: Int_, should_tabulate: Bool_) -> "DistanceTabulation":
        """Update state and optionally count the transition.

        Parameters
        ----------
        choice : Int_
            Recalled item (1-indexed study position).
        should_tabulate : Bool_
            Whether to count this transition.

        Returns
        -------
        DistanceTabulation
            Updated tabulation state.

        """

        def _update_state() -> "DistanceTabulation":
            new_previous_item = choice
            new_avail_items = self.avail_items.at[choice - 1].set(False)

            def _with_counts() -> "DistanceTabulation":
                new_actual, new_avail = self._compute_transition_counts(choice)
                return self.replace(
                    previous_item=new_previous_item,
                    avail_items=new_avail_items,
                    actual_transitions=new_actual,
                    avail_transitions=new_avail,
                )

            def _without_counts() -> "DistanceTabulation":
                return self.replace(
                    previous_item=new_previous_item,
                    avail_items=new_avail_items,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(choice > 0, _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    should_tabulate: Bool[Array, " recall_events"],
    distance_matrix: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"],
) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
    """Return actual and available bin counts for a trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events as study positions.
    present_ids : Integer[Array, " study_item_ids"]
        Item identifiers at each study position.
    should_tabulate : Bool[Array, " recall_events"]
        Boolean mask; True counts the transition.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise distances indexed by item identifier.
    bin_edges : Float[Array, " edges"]
        Interior bin edges shared across trials.

    Returns
    -------
    tuple[Integer[Array, " bins"], Integer[Array, " bins"]]
        Actual and available transition counts per bin.

    """
    valid = present_ids > 0
    remapped = jnp.where(valid, present_ids - 1, 0)
    trial_distances = distance_matrix[remapped[:, None], remapped[None, :]]
    trial_distances = jnp.where(valid[:, None] & valid[None, :], trial_distances, 0.0)

    init = DistanceTabulation(
        availability_mask=valid,
        first_recall=trial[0],
        trial_distances=trial_distances,
        bin_edges=bin_edges,
    )

    def step(tab: DistanceTabulation, idx: Int_) -> tuple[DistanceTabulation, None]:
        return tab.tabulate(trial[idx], should_tabulate[idx]), None

    result, _ = lax.scan(step, init, jnp.arange(1, trial.size))
    return result.actual_transitions, result.avail_transitions


def dist_crp(
    dataset: RecallDataset,
    distance_matrix: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"],
) -> Float[Array, " bins"]:
    """Return distance-conditioned CRP with transition filtering.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemids``, and
        ``_should_tabulate``.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise item distances.
    bin_edges : Float[Array, " edges"]
        Interior boundaries for distance bins.

    Returns
    -------
    Float[Array, " bins"]
        Conditional response probability per distance bin.

    """
    should_tabulate = jnp.asarray(dataset["_should_tabulate"], dtype=bool)

    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, 0, None, None))(
        dataset["recalls"],
        dataset["pres_itemids"],
        should_tabulate,
        distance_matrix,
        bin_edges,
    )
    return actual.sum(0) / possible.sum(0)


def plot_dist_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    should_tabulate: (
        Sequence[Bool[Array, " trial_count recall_events"]]
        | Bool[Array, " trial_count recall_events"]
    ),
    features: Float[Array, "word_count features_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    min_transitions_per_subject: int = 10,
    bin_step: float = 0.05,
    bin_source_index: int = 0,
    bin_edges: Optional[Float[Array, " edges"]] = None,
    bin_centers: Optional[Float[Array, " bins"]] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot distance-binned CRP with conditional tabulation.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to contrast.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials per dataset.
    should_tabulate : Sequence | array
        Boolean masks aligned to recall events.
    features : Float[Array, "word_count features_count"]
        Feature matrix aligned with vocabulary items.
    color_cycle : list[str], optional
        Colors for successive datasets.
    labels : Sequence[str], optional
        Legend labels for ``datasets``.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing matplotlib Axes to plot on.
    min_transitions_per_subject : int, optional
        Minimum available transitions per bin per subject.
    bin_step : float, optional
        Distance increment for expanding each bin.
    bin_source_index : int, optional
        Dataset index providing binning availability.
    bin_edges : Float[Array, " edges"], optional
        Interior bin edges; computed if ``None``.
    bin_centers : Float[Array, " bins"], optional
        Bin centers matching ``bin_edges``.
    confidence_level : float, optional
        Confidence level for the bounds.

    Returns
    -------
    Axes
        Axes with distance-binned CRP curves.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    if not isinstance(should_tabulate, Sequence):
        should_tabulate = [jnp.array(should_tabulate)]

    distances = 1 - cosine_similarity_matrix(features)

    if bin_edges is None:
        bin_edges, bin_centers = compute_distance_bins_min_count(
            datasets[bin_source_index],
            distances,
            min_transitions_per_subject=min_transitions_per_subject,
            step=bin_step,
            trial_mask=trial_masks[bin_source_index],
        )
    elif bin_centers is None:
        min_distance = jnp.min(distances)
        max_distance = jnp.max(distances)
        full_edges = jnp.concatenate(
            (min_distance[None], bin_edges, max_distance[None])
        )
        bin_centers = 0.5 * (full_edges[:-1] + full_edges[1:])

    for data_index, data in enumerate(datasets):
        data_with_mask = {
            **data,
            "_should_tabulate": should_tabulate[data_index],
        }
        subject_values = apply_by_subject(
            data_with_mask,
            trial_masks[data_index],
            jit(dist_crp),
            distances,
            bin_edges,
        )
        subject_values = jnp.vstack(subject_values)

        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            bin_centers,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(
        axis, "Semantic Distance (bin center)", "Conditional Resp. Prob.", contrast_name
    )
    return axis
