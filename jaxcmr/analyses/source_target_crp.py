"""Source-target conditional response probability.

Computes availability-adjusted transition probabilities from source
item conditions to target item conditions. This supports plots where the
x-axis is the previous recalled item's condition and separate lines show
the condition of the next recalled item.

"""

__all__ = [
    "SourceTargetTabulation",
    "tabulate_trial",
    "source_target_crp",
    "subject_source_target_crp",
    "plot_source_target_crp",
]

from typing import Optional, Sequence

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset
from .conditional_crp import base_tabulation_mask
from .crp import set_false_at_index


class SourceTargetTabulation(Pytree):
    """Per-transition state for source-target CRP."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        source_conditions: Integer[Array, " study_events"],
        target_conditions: Integer[Array, " study_events"],
        source_values: Integer[Array, " source_conditions"],
        target_values: Integer[Array, " target_conditions"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.size = size
        self.list_length = presentation.size
        self.presentation = presentation
        self.valid_targets = presentation > 0
        self.source_conditions = source_conditions
        self.target_conditions = target_conditions
        self.source_values = source_values
        self.target_values = target_values
        self.source_indices = jnp.arange(source_values.size, dtype=int)
        self.target_indices = jnp.arange(target_values.size, dtype=int)
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_counts = jnp.zeros(
            (source_values.size, target_values.size), dtype=int
        )
        self.avail_counts = jnp.zeros(
            (source_values.size, target_values.size), dtype=int
        )

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for study positions of ``recall``."""
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when recall maps to an available position."""

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            clipped_positions = jnp.clip(
                recall_study_positions - 1, 0, self.list_length - 1
            )
            is_available_study_position = self.avail_recalls[clipped_positions]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            (recall > 0) & (recall <= self.list_length),
            _for_nonzero,
            lambda: jnp.array(False),
        )

    def source_matches(self, source_value: Int_) -> Bool:
        """Return whether the previous recalled item has ``source_value``."""
        clipped_positions = jnp.clip(self.previous_positions - 1, 0, self.list_length - 1)
        is_valid_position = self.previous_positions != 0
        return jnp.any(
            is_valid_position & (self.source_conditions[clipped_positions] == source_value)
        )

    def actual_target_matches(
        self, target_value: Int_, recall_positions: Integer[Array, " size"]
    ) -> Bool:
        """Return whether current recall has ``target_value``."""
        clipped_positions = jnp.clip(recall_positions - 1, 0, self.list_length - 1)
        is_valid_position = recall_positions != 0
        is_available_position = self.avail_recalls[clipped_positions]
        return jnp.any(
            is_valid_position
            & is_available_position
            & (self.target_conditions[clipped_positions] == target_value)
        )

    def available_target_count(self, target_value: Int_) -> Int_:
        """Count currently available targets with ``target_value``."""
        is_target = self.target_conditions == target_value
        is_available = self.valid_targets & self.avail_recalls
        return jnp.sum(is_target & is_available)

    def tabulate_source_target(
        self,
        source_index: Int_,
        target_index: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> tuple[Int_, Int_]:
        """Return actual and available counts for one source-target cell."""
        source_value = self.source_values[source_index]
        target_value = self.target_values[target_index]
        source_match = self.source_matches(source_value)
        available = jnp.where(
            source_match, self.available_target_count(target_value), 0
        )
        actual = jnp.where(
            source_match & self.actual_target_matches(target_value, recall_positions),
            1,
            0,
        )
        return actual, available

    def tabulate_target_values(
        self,
        source_index: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> tuple[
        Integer[Array, " target_conditions"],
        Integer[Array, " target_conditions"],
    ]:
        """Return target counts for one source value."""
        return lax.map(
            lambda target_index: self.tabulate_source_target(
                source_index, target_index, recall_positions
            ),
            self.target_indices,
        )

    def tabulate_counts(
        self, recall: Int_
    ) -> tuple[
        Integer[Array, " source_conditions target_conditions"],
        Integer[Array, " source_conditions target_conditions"],
    ]:
        """Return cumulative source-target actual and available counts."""
        recall_positions = self.item_study_positions[recall - 1]
        actual, avail = lax.map(
            lambda source_index: self.tabulate_target_values(
                source_index, recall_positions
            ),
            self.source_indices,
        )
        return self.actual_counts + actual, self.avail_counts + avail

    def tabulate(self, recall: Int_, should_tabulate: Bool_) -> "SourceTargetTabulation":
        """Update state and optionally count the transition."""

        def _update_state() -> "SourceTargetTabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "SourceTargetTabulation":
                actual_counts, avail_counts = self.tabulate_counts(recall)
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_counts=actual_counts,
                    avail_counts=avail_counts,
                )

            def _without_counts() -> "SourceTargetTabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    source_conditions: Integer[Array, " study_events"],
    target_conditions: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    source_values: Integer[Array, " source_conditions"],
    target_values: Integer[Array, " target_conditions"],
    size: int = 3,
) -> tuple[
    Integer[Array, " source_conditions target_conditions"],
    Integer[Array, " source_conditions target_conditions"],
]:
    """Tabulate source-target actual and available counts for one trial."""
    source_values = jnp.asarray(source_values)
    target_values = jnp.asarray(target_values)
    has_recall = jnp.any(trial > 0)
    first_index = jnp.argmax(trial > 0)

    def tabulate_nonempty():
        init = SourceTargetTabulation(
            presentation,
            source_conditions,
            target_conditions,
            source_values,
            target_values,
            trial[first_index],
            size,
        )
        later_trial = jnp.where(jnp.arange(trial.size) > first_index, trial, 0)
        tab = lax.fori_loop(
            0,
            trial.size,
            lambda i, t: t.tabulate(later_trial[i], should_tabulate[i]),
            init,
        )
        return tab.actual_counts, tab.avail_counts

    def tabulate_empty():
        empty_counts = jnp.zeros((source_values.size, target_values.size), dtype=int)
        return empty_counts, empty_counts

    return lax.cond(has_recall, tabulate_nonempty, tabulate_empty)


def source_target_crp(
    dataset: RecallDataset,
    source_field: str,
    source_values: Sequence[int] | Integer[Array, " source_conditions"],
    target_field: str,
    target_values: Sequence[int] | Integer[Array, " target_conditions"],
    size: int = 3,
) -> Float[Array, " source_conditions target_conditions"]:
    """Compute source-target conditional response probability.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``, source field, and
        target field.
    source_field : str
        Dataset key used to condition on the previous recalled item.
    source_values : Sequence[int]
        Source values to tabulate on the x-axis.
    target_field : str
        Dataset key used to condition candidate target denominators.
    target_values : Sequence[int]
        Target values to plot as separate curves.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " source_conditions target_conditions"]
        Source-by-target probabilities; NaN where denominator is zero.

    """
    source_values = jnp.asarray(source_values)
    target_values = jnp.asarray(target_values)
    should_tabulate = base_tabulation_mask(dataset)

    actual, possible = vmap(
        tabulate_trial, in_axes=(0, 0, 0, 0, 0, None, None, None)
    )(
        dataset["recalls"],
        dataset["pres_itemnos"],
        dataset[source_field],  # type: ignore
        dataset[target_field],  # type: ignore
        should_tabulate,
        source_values,
        target_values,
        size,
    )
    actual_sum = actual.sum(0)
    possible_sum = possible.sum(0)
    return jnp.where(possible_sum > 0, actual_sum / possible_sum, jnp.nan)


def subject_source_target_crp(
    dataset: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    source_field: str,
    source_values: Sequence[int] | Integer[Array, " source_conditions"],
    target_field: str,
    target_values: Sequence[int] | Integer[Array, " target_conditions"],
    size: int = 3,
) -> np.ndarray:
    """Compute source-target CRP for each subject."""
    subject_values = apply_by_subject(
        dataset,
        trial_mask,
        jit(source_target_crp, static_argnames=("source_field", "target_field", "size")),
        source_field,
        jnp.asarray(source_values),
        target_field,
        jnp.asarray(target_values),
        size,
    )
    return np.asarray(subject_values)


def plot_source_target_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    source_field: str,
    source_values: Sequence[int],
    target_field: str,
    target_values: Sequence[int],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    source_labels: Optional[Sequence[str]] = None,
    xlabel: str = "Source Item",
    ylabel: str = "Avail.-Adjusted Transition Prob.",
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise source-target CRP with error bounds."""
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    target_count = len(target_values)
    if labels is None:
        labels = [""] * target_count

    x_values = jnp.arange(len(source_values))
    for data_index, data in enumerate(datasets):
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(
                source_target_crp,
                static_argnames=("source_field", "target_field", "size"),
            ),
            source_field,
            jnp.asarray(source_values),
            target_field,
            jnp.asarray(target_values),
            size,
        )
        subject_values = jnp.asarray(subject_values)

        for target_index in range(target_count):
            color_idx = data_index * target_count + target_index
            color = color_cycle[color_idx % len(color_cycle)]
            label = (
                labels[color_idx]
                if len(labels) > target_count
                else labels[target_index]
            )
            plot_data(
                axis,
                x_values,
                subject_values[:, :, target_index],
                label,
                color,
                confidence_level=confidence_level,
            )

    axis.set_xticks(x_values)
    if source_labels is not None:
        axis.set_xticklabels(source_labels)
    set_plot_labels(axis, xlabel, ylabel, contrast_name)
    return axis
