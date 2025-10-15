"""Compute distance-binned conditional response probabilities.

The module mirrors the lag-based CRP workflow but replaces lags with semantic or
spatial distances supplied by the caller. Each transition contributes to a
single distance bin, and availability is tallied as the set of bins containing
at least one unrecalled item at the moment of choice.
"""

from __future__ import annotations

__all__ = [
    "compute_distance_bin_edges",
    "compute_distance_bins_percentiles",
    "raw_candidate_transitions",
    "compute_min_count_distance_bins",
    "compute_distance_bins_min_count",
    "DistanceTabulation",
    "tabulate_trial",
    "dist_crp",
    "plot_dist_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
import numpy as np
from matplotlib import rcParams  # type: ignore
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import init_plot, plot_data, set_plot_labels
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def compute_distance_bins_percentiles(
    distance_matrix: Float[Array, " item_count item_count"],
    percentiles: Float[Array, " percentiles"],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns percentile-based distance bin edges and representative centers.

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


def compute_distance_bin_edges(
    distance_matrix: Float[Array, " item_count item_count"],
    percentiles: Float[Array, " percentiles"],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns percentile-based distance bin edges.

    Deprecated alias for :func:`compute_distance_bins_percentiles`.
    """

    return compute_distance_bins_percentiles(distance_matrix, percentiles)


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

    def _update(self, current_item: Int_) -> "DistanceTabulation":
        """Update transition tallies for the supplied recall choice."""

        distances_from_prev = self.trial_distances[self.previous_item - 1]
        actual_distance = distances_from_prev[current_item - 1]
        actual_bin = jnp.digitize(actual_distance, self.bin_edges)

        present_bins = jnp.digitize(distances_from_prev, self.bin_edges)
        bin_count = self.bin_edges.shape[0] + 1
        masked_bins = jnp.where(self.avail_items, present_bins, bin_count)
        bin_flags = jnp.zeros(bin_count + 1, dtype=jnp.int32).at[masked_bins].set(1)
        bin_flags = bin_flags[:-1]

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
    result = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return result.actual_transitions, result.avail_transitions


def raw_candidate_transitions(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    distance_matrix: Float[Array, " item_count item_count"],
) -> tuple[
    Float[Array, " transitions study_events"], Bool[Array, " transitions study_events"]
]:
    """Returns the available transition distances per recall step.

    Following the “available transitions” tally described in the semantic-CRP
    methodology, the function records, for each recall step, the distance from
    the previously recalled item to every study item that remains available.
    Callers can aggregate these distances across subjects to determine bin
    boundaries that guarantee a minimum average number of opportunities in each
    bin.

    Args:
      trial: Sequence of recall events encoded as study positions.
      present_ids: Item identifiers presented at each study position.
      distance_matrix: Pairwise item distances indexed by identifier.

    Returns:
      (distances, mask): Distance candidates and their availability mask per
      transition.
    """

    valid = present_ids > 0
    remapped = jnp.where(valid, present_ids - 1, 0)
    trial_distances = distance_matrix[remapped[:, None], remapped[None, :]]
    trial_distances = jnp.where(valid[:, None] & valid[None, :], trial_distances, 0.0)
    first_recall = trial[0]
    initial_availability = valid.at[first_recall - 1].set(False)
    positions = jnp.arange(present_ids.size)

    def step(
        carry: tuple[Int_, Bool[Array, " study_events"]],
        choice: Int_,
    ) -> tuple[
        tuple[Int_, Bool[Array, " study_events"]],
        tuple[
            Float[Array, " study_events"],
            Bool[Array, " study_events"],
        ],
    ]:
        previous_item, availability = carry
        distances_from_prev = trial_distances[previous_item - 1]
        is_valid = choice > 0
        mask = jnp.where(is_valid, availability, jnp.zeros_like(availability))
        # Candidate distances capture every potential transition emerging from
        # the current cue, mirroring the availability counts used in CRP.
        candidates = jnp.where(mask, distances_from_prev, 0.0)
        clear_mask = jnp.logical_and(is_valid, positions == (choice - 1))
        availability = jnp.where(clear_mask, False, availability)
        previous_item = jnp.where(is_valid, choice, previous_item)
        return (previous_item, availability), (candidates, mask)

    _, (distances, masks) = lax.scan(
        step,
        (first_recall, initial_availability),
        trial[1:],
    )
    return distances, masks


def compute_min_count_distance_bins(
    candidates: Float[Array, "subject_count transition_count"],
    mask: Bool[Array, "subject_count transition_count"],
    min_transitions_per_subject: int,
    step: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns distance bins that satisfy the per-subject availability minimum.

    Mirrors the semantic-CRP procedure by sweeping from the smallest distances
    (strongest similarities) upward in ``step`` increments, accumulating
    available transitions until each bin contains at least
    ``min_transitions_per_subject`` on average across subjects. The lower bound
    of the completed bin seeds the next bin, and any remaining distances form a
    final bin whose center is the mean distance of its members.

    Args:
      candidates: Distance values for every available transition per subject.
      mask: Boolean mask selecting valid entries in ``candidates``.
      min_transitions_per_subject: Minimum average number of available transitions.
      step: Increment used when widening the current bin.

    Returns:
      Tuple ``(edges, centers)`` where ``edges`` are the interior bin boundaries
      and ``centers`` are the mean distances within each bin.
    """

    subject_count = candidates.shape[0]
    required = min_transitions_per_subject * subject_count
    output_dtype = jnp.asarray(candidates).dtype
    distances = np.asarray(candidates)[np.asarray(mask)]
    distances.sort()

    bins: list[tuple[float, float, float]] = []
    max_distance = distances[-1]
    index = 0
    current_lower = distances[0]
    total_count = distances.size

    while index < total_count:
        current_upper = current_lower
        start = index

        while index < total_count and distances[index] <= current_upper:
            index += 1

        while (index - start) < required and current_upper < max_distance:
            current_upper = min(current_upper + step, max_distance)
            while index < total_count and distances[index] <= current_upper:
                index += 1

        bin_slice = distances[start:index]
        bins.append((current_lower, current_upper, float(bin_slice.mean())))
        current_lower = current_upper

    interior_edges = jnp.asarray([each[1] for each in bins[:-1]], dtype=output_dtype)
    centers = jnp.asarray([each[2] for each in bins], dtype=output_dtype)
    return interior_edges, centers


def compute_distance_bins_min_count(
    dataset: RecallDataset,
    distance_matrix: Float[Array, " item_count item_count"],
    min_transitions_per_subject: int,
    step: float = 0.05,
    trial_mask: Optional[Bool[Array, " trial_count"]] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns distance bins derived from availability counts.

    The function gathers the available transition distances across the dataset
    and applies ``compute_min_count_distance_bins`` so that each distance bin
    contains, on average, at least ``min_transitions_per_subject`` transition
    opportunities per subject. This mirrors the semantic-CRP binning rule that
    decrements the lower distance boundary by roughly 0.05 until the required
    availability is met.

    Args:
      dataset: Recall dataset with ``recalls`` and ``pres_itemids`` fields.
      distance_matrix: Pairwise distances indexed by item identifier.
      min_transitions_per_subject: Minimum number of available transitions per
        bin, averaged across subjects.
      step: Distance increment used while expanding each bin.
      trial_mask: Optional mask selecting which trials contribute to the bin
        definition.

    Returns:
      Tuple ``(edges, centers)`` where ``edges`` are the interior boundaries and
      ``centers`` are the mean distances per bin.
    """

    recalls = dataset["recalls"]
    if trial_mask is None:
        trial_mask = jnp.ones(recalls.shape[0], dtype=bool)
    else:
        trial_mask = jnp.asarray(trial_mask, dtype=bool)

    def gather_subject(
        subject_dataset: RecallDataset, matrix: Float[Array, " item_count item_count"]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        distances, masks = vmap(raw_candidate_transitions, in_axes=(0, 0, None))(
            subject_dataset["recalls"],
            subject_dataset["pres_itemids"],
            matrix,
        )
        distances = distances.reshape(-1)
        masks = masks.reshape(-1)
        return distances, masks

    subject_outputs = apply_by_subject(
        dataset,
        trial_mask,
        gather_subject,
        distance_matrix,
    )
    if not subject_outputs:
        raise ValueError(
            "No available transition distances remain after applying the trial mask."
        )
    subject_distances = jnp.vstack([each[0] for each in subject_outputs])
    subject_masks = jnp.vstack([each[1] for each in subject_outputs])
    return compute_min_count_distance_bins(
        subject_distances,
        subject_masks,
        min_transitions_per_subject,
        step,
    )


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
        dataset["recalls"],
        dataset["pres_itemids"],
        distance_matrix,
        bin_edges,
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
    min_transitions_per_subject: int = 10,
    bin_step: float = 0.05,
    bin_source_index: int = 0,
) -> Axes:
    """Plot distance-binned CRP curves aggregated by subject.

    Args:
      datasets: Collection of recall datasets to contrast.
      trial_masks: Boolean masks selecting trials per dataset.
      distances: Distance matrix (or matrices) defining the metric.
      color_cycle: Colors used for successive datasets.
      labels: Legend labels corresponding to ``datasets``.
      contrast_name: Optional legend title.
      axis: Optional matplotlib axis to draw on.
      size: Unused, included for swappability with other plotting functions.
      min_transitions_per_subject: Minimum number of available transitions per
        bin, averaged across subjects, used when defining distance bins.
      bin_step: Distance increment applied while expanding each bin.
      bin_source_index: Index selecting which dataset provides the binning
        availability counts.
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

    bin_edges, bin_centers = compute_distance_bins_min_count(
        datasets[bin_source_index],
        distances,
        min_transitions_per_subject=min_transitions_per_subject,
        step=bin_step,
        trial_mask=trial_masks[bin_source_index],
    )

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
