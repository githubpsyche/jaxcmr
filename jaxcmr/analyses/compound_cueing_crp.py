"""Compound cueing conditional response probability.

Tests whether recall of a repeated item is influenced by the two most
recent recalls jointly (compound cue) rather than just the immediately
preceding item. Classifies each transition to a repeated item as
"pure" (both prior recalls neighbor the same presentation) or "mixed"
(prior recalls neighbor different presentations).

Notes
-----
- Tracks a two-item recall history (``prev_position`` and
  ``prev_prev_position``) to classify the cueing pattern.
- Counts are stored as int [4]: ``[pure_actual, pure_avail,
  mixed_actual, mixed_avail]``.
- Only transitions TO repeated items with sufficient spacing
  (``min_spacing``) are counted.

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
    """Set ``vec[i - 1]`` to ``False`` (1-based; 0 is a no-op).

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean vector.
    i : Int_
        1-based index; 0 is a no-op sentinel.

    Returns
    -------
    tuple
        Updated vector and ``None``.

    """
    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class CompoundCueingTabulation(Pytree):
    """Tabulate compound cueing transitions to repeated items.

    Parameters
    ----------
    presentation : Integer[Array, " study_events"]
        Item numbers at each study position (1-indexed).
    first_recall : Int_
        First recalled item number (1-indexed).
    min_spacing : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.

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
        """Check cueing conditions for one item and return updates.

        Parameters
        ----------
        item_idx : Int_
            0-indexed item position.
        recall : Int_
            Currently recalled item (1-indexed).

        Returns
        -------
        Integer[Array, "4"]
            Counts [pure_actual, pure_avail, mixed_actual,
            mixed_avail].

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

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recalled item numbers (1-indexed; 0 = padding).
    presentation : Integer[Array, " study_events"]
        Item numbers at each study position.
    min_spacing : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    Integer[Array, "4"]
        Counts [pure_actual, pure_avail, mixed_actual,
        mixed_avail].

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

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    min_spacing : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    Float[Array, "2"]
        [pure_crp, mixed_crp]; NaN where no opportunities.

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
    """Plot compound cueing as a grouped bar chart.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    min_spacing : int
        Minimum spacing between repeated occurrences.
    size : int
        Maximum positions an item can occupy.
    color_cycle : list[str], optional
        Colors for each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Axes to plot on.

    Returns
    -------
    Axes
        Axes with bar plot comparing pure vs mixed cueing.

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
