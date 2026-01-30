"""
Compound Cueing Conditional Response Probability.

Overview:
    Utilities to compute and plot compound cueing effects for transitions to
    repeated items. This analysis tests a prediction that differentiates
    composite memory models (CMR) from instance-based models (ICMR).

Background:
    For a repeated item at study positions i and j (with sufficient spacing),
    the compound cueing analysis examines how the *two most recent* recalls
    influence the probability of transitioning to the repeated item.

    - "Pure" cueing: last two recalls are {i-2, i-1} or {j-2, j-1} (both from
      one occurrence's neighborhood)
    - "Mixed" cueing: last two recalls are {j-2, i-1} or {i-2, j-1} (one from
      each occurrence's neighborhood)

Theoretical predictions:
    - CMR (tau=1): Mixed cueing provides equal or greater support because
      similarities sum linearly: moderate + moderate ≈ high + low.
    - ICMR (tau>1): Pure cueing provides greater support because sharpening
      happens before summing: high^τ > moderate^τ + moderate^τ when τ > 1.

Definition:
    P(pure) = actual_pure / available_pure
    P(mixed) = actual_mixed / available_mixed

    Where:
    - actual_{type} = count of transitions to repeated items after {type} cueing
    - available_{type} = count of opportunities where repeated item was available

Conventions:
    - Study positions are 1-indexed; 0 indicates padding.
    - min_spacing: Minimum separation between repeated item occurrences to be
      included in analysis (default 6).
    - size: Maximum number of study positions an item can occupy (default 2).

Design decisions:
    - Padding & bounds: Zeros and invalid recalls are ignored via guards.
    - Division by zero: Conditions with zero availability yield NaN.
    - Repeat tracking: Uses pre-computed item_study_positions for efficiency.

JAX compilation:
    - All functions are side-effect-free and JIT-safe.
    - Use `jit(compound_cueing_crp, static_argnames=("min_spacing", "size"))`.
    - Keep shapes consistent within a compiled call to avoid recompiles.
"""

__all__ = [
    "CompoundCueingTabulation",
    "tabulate_trial",
    "compound_cueing_crp",
    "plot_compound_cueing",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import prepare_plot_inputs
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Indices are 1-based; ``0`` is a no-op sentinel. Indices outside
    ``[1, vec.size]`` are ignored.

    Returns:
        Tuple of the (possibly updated) vector and ``None``.
    """
    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class CompoundCueingTabulation(Pytree):
    """Tabulation of compound cueing effects for transitions to repeated items.

    Tracks two-item recall history to detect pure ({i-2, i-1} or {j-2, j-1})
    and mixed ({j-2, i-1} or {i-2, j-1}) patterns, then tabulates transitions
    to repeated items.

    State:
        - prev_position, prev_prev_position: last two recalled study positions
        - avail_recalls: boolean [L], study positions still available
        - counts: int [4], accumulated [pure_actual, pure_avail, mixed_actual, mixed_avail]

    Args:
        presentation: Item numbers at each study position (1-indexed; 0 = padding).
        first_recall: First recalled item number (1-indexed).
        min_spacing: Minimum spacing between repeated item occurrences.
        size: Maximum number of positions an item can occupy.
    """

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        min_spacing: int = 6,
        size: int = 2,
    ):
        self.size = size
        self.min_spacing = min_spacing
        self.list_length = presentation.size
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)

        # For each item, get all study positions (padded with zeros)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        # Pre-compute item indices for iteration (avoids tracing issues in lax.map)
        self.item_indices = jnp.arange(self.list_length, dtype=int)

        # Track which positions are still available for recall
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

        # Track last two recalled positions (most recent last)
        first_positions = self.item_study_positions[first_recall - 1]
        self.prev_prev_position = jnp.array(0, dtype=int)  # Not yet set
        self.prev_position = first_positions[0]  # Use first study position

        # Tabulation counts: [pure_actual, pure_avail, mixed_actual, mixed_avail]
        self.counts = jnp.zeros(4, dtype=int)

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for all study positions of ``recall``."""
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def is_repeater_available(self, item_idx: Int_) -> Bool:
        """Check if a repeated item (0-indexed) is available for recall."""
        positions = self.item_study_positions[item_idx]
        first_pos = positions[0]
        return lax.cond(
            first_pos > 0,
            lambda: self.avail_recalls[first_pos - 1],
            lambda: jnp.array(False),
        )

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return ``True`` when recall positions have not been retrieved yet."""
        recall_study_positions = self.item_study_positions[recall - 1]
        is_valid_study_position = recall_study_positions != 0
        is_available_study_position = self.avail_recalls[recall_study_positions - 1]
        return jnp.any(is_valid_study_position & is_available_study_position)

    def check_and_count_item(self, item_idx: Int_, recall: Int_) -> Integer[Array, "4"]:
        """Check cueing conditions for one item and return count updates.

        Returns:
            Array of [pure_actual, pure_avail, mixed_actual, mixed_avail].
        """
        positions = self.item_study_positions[item_idx]
        i_pos = positions[0]  # First occurrence position
        j_pos = positions[1]  # Second occurrence position (0 if not repeated)

        # Validity checks
        is_repeated = j_pos > 0
        has_spacing = (j_pos - i_pos) >= self.min_spacing
        is_available = self.is_repeater_available(item_idx)
        current_is_this_item = recall == (item_idx + 1)  # recall is 1-indexed
        have_two_recalls = self.prev_prev_position > 0

        valid = is_repeated & has_spacing & have_two_recalls

        # Pure cueing: {i-2, i-1} or {j-2, j-1}
        pure_near_i = (self.prev_position == (i_pos - 1)) & (
            self.prev_prev_position == (i_pos - 2)
        )
        pure_near_j = (self.prev_position == (j_pos - 1)) & (
            self.prev_prev_position == (j_pos - 2)
        )
        pure = pure_near_i | pure_near_j

        # Mixed cueing: {j-2, i-1} or {i-2, j-1}
        mixed_1 = (self.prev_position == (i_pos - 1)) & (
            self.prev_prev_position == (j_pos - 2)
        )
        mixed_2 = (self.prev_position == (j_pos - 1)) & (
            self.prev_prev_position == (i_pos - 2)
        )
        mixed = mixed_1 | mixed_2

        # Compute count updates
        pure_actual = (valid & pure & is_available & current_is_this_item).astype(int)
        pure_avail = (valid & pure & is_available).astype(int)
        mixed_actual = (valid & mixed & is_available & current_is_this_item).astype(int)
        mixed_avail = (valid & mixed & is_available).astype(int)

        return jnp.array(
            [pure_actual, pure_avail, mixed_actual, mixed_avail], dtype=int
        )

    def _do_tabulate(self, recall: Int_) -> "CompoundCueingTabulation":
        """Internal tabulation logic."""
        # Check all items and sum counts
        check_fn = lambda idx: self.check_and_count_item(idx, recall)
        all_counts = lax.map(check_fn, self.item_indices)
        new_counts = self.counts + jnp.sum(all_counts, axis=0)

        # Update recall history: prev_prev <- prev, prev <- current
        recall_positions = self.item_study_positions[recall - 1]
        new_prev_position = recall_positions[0]

        return self.replace(
            counts=new_counts,
            prev_prev_position=self.prev_position,
            prev_position=new_prev_position,
            avail_recalls=self.available_recalls_after(recall),
        )

    def tabulate(self, recall: Int_) -> "CompoundCueingTabulation":
        """Update tabulation for a recall event.

        Ignores zero-padded recalls and already-recalled items.
        """
        return lax.cond(
            self.is_valid_recall(recall),
            lambda: self._do_tabulate(recall),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    min_spacing: int = 6,
    size: int = 2,
) -> Integer[Array, "4"]:
    """Tabulate compound cueing counts for a single trial.

    Args:
        trial: Recalled item numbers (1-indexed; 0 = padding).
        presentation: Item numbers at each study position.
        min_spacing: Minimum spacing between repeated item occurrences.
        size: Maximum number of positions an item can occupy.

    Returns:
        Array of [pure_actual, pure_avail, mixed_actual, mixed_avail] counts.
    """
    init = CompoundCueingTabulation(presentation, trial[0], min_spacing, size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.counts


def compound_cueing_crp(
    dataset: RecallDataset,
    min_spacing: int = 6,
    size: int = 2,
) -> Float[Array, "2"]:
    """Compute compound cueing conditional response probabilities.

    Args:
        dataset: Recall dataset with ``recalls`` and ``pres_itemnos``.
        min_spacing: Minimum spacing between repeated item occurrences.
        size: Maximum number of positions an item can occupy.

    Returns:
        Array of [pure_crp, mixed_crp] - conditional probabilities of
        transitioning to repeated item given pure vs mixed cueing.

    Notes:
        - Pure cueing: last two recalls = {i-2, i-1} or {j-2, j-1}
        - Mixed cueing: last two recalls = {j-2, i-1} or {i-2, j-1}
        - NaN returned if no opportunities for a condition.
    """
    trials = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    counts = vmap(tabulate_trial, in_axes=(0, 0, None, None))(
        trials, presentations, min_spacing, size
    )

    # Sum across trials
    total_counts = counts.sum(axis=0)

    # Compute CRPs: actual / available (NaN if no opportunities)
    pure_crp = total_counts[0] / total_counts[1]
    mixed_crp = total_counts[2] / total_counts[3]

    return jnp.array([pure_crp, mixed_crp])


def plot_compound_cueing(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    min_spacing: int = 6,
    size: int = 2,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
) -> Axes:
    """Plot compound cueing analysis as grouped bar chart.

    Args:
        datasets: Datasets containing trial data.
        trial_masks: Masks to filter trials.
        min_spacing: Minimum spacing between repeated item occurrences.
        size: Maximum number of positions an item can occupy.
        color_cycle: Colors for plotting each dataset.
        labels: Dataset labels for legend.
        contrast_name: Name of contrast for legend labeling.
        axis: Existing Matplotlib ``Axes`` to plot on.

    Returns:
        Matplotlib Axes with bar plot comparing pure vs mixed cueing.

    Notes:
        - Error bars show standard error of the mean across subjects.
        - ICMR predicts pure > mixed; CMR predicts mixed >= pure.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    x_labels = ["Pure\n{i-2, i-1}", "Mixed\n{j-2, i-1}"]
    x_positions = jnp.array([0, 1])

    for data_index, data in enumerate(datasets):
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(compound_cueing_crp, static_argnames=("min_spacing", "size")),
            min_spacing=min_spacing,
            size=size,
        )

        # Stack subject values: shape (n_subjects, 2)
        subject_array = jnp.vstack(subject_values)

        color = color_cycle[data_index % len(color_cycle)]

        # Plot bars with error bars
        means = jnp.nanmean(subject_array, axis=0)
        n_valid = jnp.sum(~jnp.isnan(subject_array[:, 0]))
        sems = jnp.nanstd(subject_array, axis=0) / jnp.sqrt(n_valid)

        width = 0.35
        offset = (data_index - len(datasets) / 2 + 0.5) * width

        axis.bar(
            x_positions + offset,
            means,
            width,
            yerr=sems,
            label=labels[data_index] if labels[data_index] else None,
            color=color,
            capsize=3,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(x_labels)
    axis.set_ylabel("P(transition to repeated item)")

    if contrast_name:
        axis.legend(title=contrast_name)

    return axis
