"""Conditional response probability by distance between recalled items.

Given the previous recall, which distances were available for the next
recall, and which distance did the participant choose? ``tabulate_trial``
uses the items' global identifiers in ``pres_itemids`` to look up distances
in a vocabulary matrix. A transition contributes one opportunity to every
bin containing an available item, even if several items share a bin. The
chosen item's bin also receives one actual transition.

``dist_crp`` pools these counts across trials and divides within each bin.
Use common bin edges when comparing curves. Edges can come from percentiles
of the matrix or from distances to available items on selected trials. For
the latter, the target bin count is the requested minimum multiplied by the
number of participants. Counts are pooled, so participants can contribute
unequally and the final bin can fall short.
"""

from __future__ import annotations

from typing import Literal, cast

__all__ = [
    "DistanceTabulation",
    "compute_distance_bins_min_count",
    "compute_distance_bins_percentiles",
    "compute_min_count_distance_bins",
    "dist_crp",
    "plot_cat_crp",
    "plot_dist_crp",
    "raw_candidate_transitions",
    "tabulate_trial",
]

from collections.abc import Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes

from jaxcmr._vendor.simple_pytree import Pytree
from jaxcmr.dataset import _require_pres_itemids, apply_by_subject

from ..math import cosine_similarity_matrix
from ..plotting import format_analysis_axes, plot_mean_curve, prepare_plot_inputs
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def compute_distance_bins_percentiles(
    distance_matrix: Float[Array, " item_count item_count"],
    percentiles: Float[Array, " percentiles"],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Percentile-based distance bin edges and centers.

    Parameters
    ----------
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise distances for the item vocabulary.
    percentiles : Float[Array, " percentiles"]
        Percentiles (0--100) defining interior bin edges.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(interior_edges, bin_centers)`` where edges are the
        percentile cut points and centers are bin midpoints.

    """

    matrix = np.asarray(distance_matrix)
    upper_indices = np.triu_indices(matrix.shape[0], k=1)
    upper_values = matrix[upper_indices]
    upper_values = upper_values[np.isfinite(upper_values)]
    if upper_values.size == 0:
        raise ValueError("No finite distance values found.")
    output_dtype = jnp.asarray(distance_matrix).dtype
    interior = jnp.asarray(
        np.percentile(upper_values, np.asarray(percentiles)), dtype=output_dtype
    )
    full_edges = jnp.concatenate(
        (
            jnp.asarray([upper_values.min()], dtype=output_dtype),
            interior,
            jnp.asarray([upper_values.max()], dtype=output_dtype),
        )
    )
    centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    return interior, centers


class DistanceTabulation(Pytree):
    """Accumulates per-bin transition counts for a single trial."""

    def __init__(
        self,
        availability_mask: Bool[Array, " study_events"],
        first_recall: Int_,
        trial_distances: Float[Array, "study_events study_events"],
        bin_edges: Float[Array, " edges"],
    ) -> None:
        bin_count = bin_edges.shape[0] + 1
        self.bin_edges = bin_edges
        self.trial_distances = trial_distances
        self.actual_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_transitions = jnp.zeros(bin_count, dtype=jnp.int32)
        self.avail_items = availability_mask
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall

    def _update(self, current_item: Int_) -> DistanceTabulation:
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

    def tabulate(self, choice: Int_) -> DistanceTabulation:
        "Tabulate a transition if the choice is non-zero (i.e., a valid item)."
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    distance_matrix: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"],
) -> tuple[Integer[Array, " bins"], Integer[Array, " bins"]]:
    """Actual and available bin counts for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events encoded as study positions.
    present_ids : Integer[Array, " study_item_ids"]
        Item identifiers at each study position.
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
    result = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return result.actual_transitions, result.avail_transitions


def raw_candidate_transitions(
    trial: Integer[Array, " recall_events"],
    present_ids: Integer[Array, " study_item_ids"],
    distance_matrix: Float[Array, " item_count item_count"],
) -> tuple[
    Float[Array, " transitions study_events"], Bool[Array, " transitions study_events"]
]:
    """Available transition distances per recall step.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events encoded as study positions.
    present_ids : Integer[Array, " study_item_ids"]
        Item identifiers at each study position.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise item distances indexed by identifier.

    Returns
    -------
    tuple
        ``(distances, mask)`` with distance candidates and
        their availability mask per transition.

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
    """Distance bins targeting a pooled count of available alternatives.

    Parameters
    ----------
    candidates : Float[Array, "subject_count transition_count"]
        Distance values for every available transition.
    mask : Bool[Array, "subject_count transition_count"]
        Boolean mask selecting valid entries in ``candidates``.
    min_transitions_per_subject : int
        Minimum average available transitions per bin.
    step : float
        Increment used when widening the current bin.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(edges, centers)`` with interior bin boundaries and
        mean distances per bin.

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
    trial_mask: Bool[Array, " trial_count"] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Distance bins derived from dataset availability counts.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemids``.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise distances indexed by item identifier.
    min_transitions_per_subject : int
        Target average number of available alternatives per participant per bin.
    step : float
        Distance increment used while expanding each bin.
    trial_mask : Bool[Array, " trial_count"], optional
        Mask selecting trials for bin computation.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(edges, centers)`` with interior boundaries and
        mean distances per bin.

    """

    recalls = dataset["recalls"]
    if trial_mask is None:
        trial_mask = jnp.ones(recalls.shape[0], dtype=bool)

    def gather_subject(
        subject_dataset: RecallDataset, matrix: Float[Array, " item_count item_count"]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        distances, masks = vmap(raw_candidate_transitions, in_axes=(0, 0, None))(
            subject_dataset["recalls"],
            _require_pres_itemids(subject_dataset),
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
    """Distance-conditioned response probability.

    This numerical operation is JIT-compatible and requires aligned integer
    item IDs within matrix coverage. Supply a square matrix with finite distances
    for the within-list alternatives used by the analysis. Input alignment and
    distance suitability are the caller's responsibility.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemids``.
    distance_matrix : Float[Array, " item_count item_count"]
        Pairwise item distances.
    bin_edges : Float[Array, " edges"]
        Interior boundaries for distance bins.

    Returns
    -------
    Float[Array, " bins"]
        Conditional response probability per distance bin.

    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        dataset["recalls"],
        _require_pres_itemids(dataset),
        distance_matrix,
        bin_edges,
    )
    return actual.sum(0) / possible.sum(0)


def _resolve_distance_bins(
    datasets: Sequence[RecallDataset],
    trial_masks: Sequence[Bool[Array, " trial_count"]],
    distances: Float[Array, " item_count item_count"],
    bin_edges: Float[Array, " edges"] | Literal["min_count", "percentile"] | None,
    bin_centers: Float[Array, " bins"] | None,
    min_transitions_per_subject: int,
    bin_step: float,
    bin_source_index: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve named or explicit bin edges for plot wrappers."""
    if bin_edges is None:
        bin_edges = "percentile"

    if isinstance(bin_edges, str):
        if bin_edges == "min_count":
            return compute_distance_bins_min_count(
                datasets[bin_source_index],
                distances,
                min_transitions_per_subject=min_transitions_per_subject,
                step=bin_step,
                trial_mask=trial_masks[bin_source_index],
            )
        return compute_distance_bins_percentiles(
            distances,
            jnp.linspace(1, 99, 10),
        )

    bin_edges = jnp.asarray(bin_edges)
    if bin_centers is None:
        finite_distances = np.asarray(distances)
        finite_distances = finite_distances[np.isfinite(finite_distances)]
        if finite_distances.size == 0:
            raise ValueError("No finite distance values found.")
        output_dtype = jnp.asarray(distances).dtype
        full_edges = jnp.concatenate(
            (
                jnp.asarray([finite_distances.min()], dtype=output_dtype),
                bin_edges,
                jnp.asarray([finite_distances.max()], dtype=output_dtype),
            )
        )
        bin_centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    return bin_edges, jnp.asarray(bin_centers)


def plot_dist_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    features: Float[Array, "word_count features_count"] | None = None,
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    min_transitions_per_subject: int = 10,
    bin_step: float = 0.05,
    bin_source_index: int = 0,
    bin_edges: Float[Array, " edges"]
    | Literal["min_count", "percentile"]
    | None = "percentile",
    bin_centers: Float[Array, " bins"] | None = None,
    distance_matrix: Float[Array, " item_count item_count"] | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot distance-binned CRP curves aggregated by subject.

    Supply features or a distance matrix. When both are supplied, the distance
    matrix takes precedence.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to contrast.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials per dataset.
    features : Float[Array, "word_count features_count"], optional
        Feature matrix whose rows align with vocabulary items.
    color_cycle : list[str], optional
        Colors for successive datasets.
    labels : Sequence[str], optional
        Legend labels for ``datasets``.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Matplotlib Axes to plot on.
    min_transitions_per_subject : int
        Target average number of available alternatives per participant per bin.
    bin_step : float
        Distance increment for expanding each bin.
    bin_source_index : int
        Dataset index providing binning availability counts.
    bin_edges : {"percentile", "min_count"} or Float[Array, " edges"], optional
        Named binning rule or explicit interior bin edges. ``None`` uses
        ``"percentile"``.
    bin_centers : Float[Array, " bins"], optional
        Bin centers matching ``bin_edges``.
    distance_matrix : Float[Array, " item_count item_count"], optional
        Pairwise distance matrix indexed by item identifier.
    confidence_level : float
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

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

    if distance_matrix is None:
        # The caller supplies features when no distance matrix is given.
        feature_matrix = cast(Float[Array, " word_count features_count"], features)
        distances = 1 - cosine_similarity_matrix(feature_matrix)
    else:
        distances = jnp.asarray(distance_matrix)

    bin_edges, bin_centers = _resolve_distance_bins(
        datasets,
        trial_masks,
        distances,
        bin_edges,
        bin_centers,
        min_transitions_per_subject,
        bin_step,
        bin_source_index,
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

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            bin_centers,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    format_analysis_axes(
        axis, "Distance (bin center)", "Conditional Resp. Prob.", contrast_name
    )
    return axis


def plot_cat_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    features: Float[Array, "word_count features_count"],
    feature_column: int = 0,
    feature_label: str = "Category",
    color_cycle: list[str] | None = None,
    labels: Sequence[str] | None = None,
    contrast_name: str | None = None,
    axis: Axes | None = None,
    confidence_level: float = 0.95,
    *,
    n_resamples: int = 9999,
    bootstrap_seed: int | None = None,
) -> Axes:
    """Plot category-binned CRP curves from a single feature column.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to compare.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials per dataset.
    features : Float[Array, "word_count features_count"]
        Feature matrix; specified column encodes categories.
    feature_column : int
        Zero-based column index in ``features``.
    feature_label : str
        Human-readable label for the feature column.
    color_cycle : list[str], optional
        Colors for each dataset.
    labels : Sequence[str], optional
        Legend labels for ``datasets``.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Matplotlib Axes to plot on.
    confidence_level : float
        Confidence level for error bounds.

    n_resamples : int, optional
        Number of BCa bootstrap resamples for participant means, default 9999.
    bootstrap_seed : int, optional
        Seed for reproducible intervals. Fewer than two finite participant scores
        gives no interval; a constant sample gives a point interval.

    Returns
    -------
    Axes
        Axes with category-binned CRP curves.
    """

    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    feature_values = jnp.asarray(
        features if features.ndim == 1 else features[:, feature_column],
        dtype=jnp.float32,
    )
    distances = jnp.abs(feature_values[:, None] - feature_values[None, :])

    bin_edges = jnp.array([0.5], dtype=jnp.float32)
    bin_centers = jnp.array([0.0, 1.0], dtype=jnp.float32)

    for data_index, data in enumerate(datasets):
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(dist_crp),
            distances,
            bin_edges,
        )
        subject_values = jnp.vstack(subject_values)

        color = color_cycle[data_index % len(color_cycle)]
        plot_mean_curve(
            axis,
            bin_centers,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    axis.set_xticks(np.array(bin_centers))
    axis.set_xticklabels(["Same", "Different"])

    format_analysis_axes(
        axis,
        f"{feature_label}",
        "Conditional Resp. Prob.",
        contrast_name,
    )
    return axis
