"""Compute distance-binned conditional response probabilities.

The module mirrors the lag-based CRP workflow but replaces lags with semantic or
spatial distances supplied by the caller. Each transition contributes to a
single distance bin, and availability is tallied as the set of bins containing
at least one unrecalled item at the moment of choice.
"""

from __future__ import annotations

__all__ = [
    "compute_distance_bin_edges",
    "DistanceTabulation",
    "tabulate_trial",
    "crp",
    "plot_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp, nn
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset
from ..helpers import apply_by_subject


def compute_distance_bin_edges(
    distance_matrix: Float[Array, " item_count item_count"],
    percentiles: Float[Array, " percentiles"],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return interior distance bin edges and representative bin centers.

    Args:
      distance_matrix: Pairwise distances for the item vocabulary.
      percentiles: Percentiles (0–100) that define interior bin edges.

    Returns:
      Tuple ``(interior_edges, bin_centers)`` where ``interior_edges`` contains
      the percentile cut points and ``bin_centers`` provides the midpoint of the
      corresponding distance ranges.
    """

    upper_indices = jnp.triu_indices(distance_matrix.shape[0], k=1)
    upper_values = distance_matrix[upper_indices]
    interior = jnp.percentile(upper_values, percentiles)
    full_edges = jnp.concatenate(
        (upper_values.min()[None], interior, upper_values.max()[None])
    )
    centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    return interior, centers


class DistanceTabulation(Pytree):
    """Accumulates per-bin transition counts for a single trial.

    Assumes:
        - ``trial[0] > 0`` (first recall exists).
        - ``presentation`` supplies item identifiers compatible with the global
          ``distance_matrix``.
        - Subsequent zeros in ``trial`` are padding and ignored.

    Behavior:
        - Tracks availability per study position, clearing positions as they are
          recalled.
        - Computes distances between the previously recalled item and each
          unrecalled study item before binning.
    """

    def __init__(
        self,
        present_ids: Integer[Array, " study_events"],
        first_recall: Int_,
        distance_matrix: Float[Array, " item_count item_count"],
        bin_edges: Float[Array, " edges"],
    ):
        bin_count = bin_edges.shape[0] + 1
        self.bin_edges = bin_edges
        self.present_ids = present_ids
        self.distance_matrix = distance_matrix
        self.actual_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_items = self.present_ids > 0
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall

    def _update(self, current_item: Int_) -> "DistanceTabulation":
        """Update transition tallies for the supplied recall choice."""

        previous_id = self.present_ids[self.previous_item - 1]
        current_id = self.present_ids[current_item - 1]
        distances_from_prev = self.distance_matrix[previous_id - 1]
        actual_distance = distances_from_prev[current_id - 1]
        actual_bin = jnp.digitize(actual_distance, self.bin_edges)
        present_distances = distances_from_prev[self.present_ids - 1]
        present_bins = jnp.digitize(present_distances, self.bin_edges)
        one_hot_bins = nn.one_hot(present_bins, self.bin_edges.shape[0] + 1)
        available_bins = jnp.logical_and(one_hot_bins, self.avail_items[:, None])
        bin_flags = jnp.any(available_bins, axis=0).astype(jnp.int32)

        return self.replace(
            previous_item=current_item,
            avail_items=self.avail_items.at[current_item - 1].set(False),
            avail_transitions=self.avail_transitions + bin_flags,
            actual_transitions=self.actual_transitions.at[actual_bin].add(1),
        )

    def tabulate(self, choice: Int_) -> "DistanceTabulation":
        "Tabulate a transition if the choice is non-zero (i.e., a valid item)."
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    distance_matrix: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"],
) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
    """Return actual and available bin counts for a single trial.

    Args:
      trial: Sequence of recall events encoded as study positions.
      present_ids: Item identifiers presented at each study position.
      distance_matrix: Pairwise distances indexed by item identifier.
      bin_edges: Interior bin edges shared across trials.
    """

    init = DistanceTabulation(present_ids, trial[0], distance_matrix, bin_edges)
    result = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return result.actual_transitions, result.avail_transitions


def dist_crp(
    dataset: RecallDataset,
    distance_matrix: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"],
) -> Float[Array, " bins"]:
    """Return the distance-conditioned response probability.

    Args:
      dataset: recall dataset containing at least ``recalls`` and ``pres_itemids``
      distance_matrix: Pairwise item distances.
      bin_edges: Interior boundaries for distance bins.
    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        dataset["recalls"], dataset["pres_itemids"], distance_matrix, bin_edges
    )
    return actual.sum(0) / possible.sum(0)


def plot_dist_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    distances: Float[Array, "word_count word_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: Optional[int] = None,
) -> Axes:
    """Plot distance-binned CRP curves aggregated by subject.

    Args:
      datasets: Collection of recall datasets to contrast.
      trial_masks: Boolean masks selecting trials per dataset.
      distances: Distance matrix (or matrices) defining the metric.
      percentiles: Percentiles used to derive bin edges when unspecified.
      color_cycle: Colors used for successive datasets.
      labels: Legend labels corresponding to ``datasets``.
      contrast_name: Optional legend title.
      axis: Optional matplotlib axis to draw on.
      size: Unused, included for swappability with other plotting functions.
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

    percentiles = jnp.linspace(1, 99, 10)
    bin_edges, bin_centers = compute_distance_bin_edges(distances, percentiles)

    for data_index, data in enumerate(datasets):
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(dist_crp),
            distances,
            bin_edges,
        )
        subject_values = jnp.vstack(subject_values)

        color = color_cycle.pop(0)
        plot_data(
            axis,
            bin_centers,
            subject_values,
            labels[data_index],
            color,
        )

    set_plot_labels(
        axis, "Distance (bin center)", "Conditional Resp. Prob.", contrast_name
    )
    return axis
