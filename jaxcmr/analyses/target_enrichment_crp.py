"""Lag-CRP target enrichment.

Computes whether recalled targets at each lag are enriched for a target
condition relative to the available targets at that same lag. This is a
CRP-like diagnostic with a zero baseline: positive values mean the actual
transitions at a lag select the target value more often than expected
from lag-specific availability.

"""

__all__ = [
    "set_false_at_index",
    "TargetEnrichmentTabulation",
    "tabulate_trial",
    "target_enrichment_crp",
    "plot_target_enrichment_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Bool_, Float, Int_, Integer, RecallDataset
from .conditional_crp import base_tabulation_mask, source_tabulation_masks
from .crp import set_false_at_index


class TargetEnrichmentTabulation(Pytree):
    """Per-transition state for lag-specific target enrichment."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        target_conditions: Integer[Array, " study_events"],
        target_value: Int_,
        target_values: Integer[Array, " target_values"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.size = size
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.presentation = presentation
        self.target_conditions = target_conditions
        self.target_value = target_value
        self.target_values = target_values
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_target_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.actual_total_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_target_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_total_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

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

    def condition_in_pool(self, position_index: Int_) -> Bool:
        """Return whether a target condition is in the comparison pool."""
        return jnp.any(self.target_conditions[position_index] == self.target_values)

    def lags_from_previous_to_target(
        self, target_pos: Int_, valid_target: Bool
    ) -> Bool[Array, " lags"]:
        """Return ordinary transition lags from previous recalls to target."""

        def f(prev):
            return lax.cond(
                valid_target & (target_pos > 0) & (prev > 0),
                lambda: self.base_lags.at[
                    target_pos - prev + self.lag_range
                ].add(1),
                lambda: self.base_lags,
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_target_lags(
        self,
        target_pos: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> tuple[
        Integer[Array, " lags"],
        Integer[Array, " lags"],
        Integer[Array, " lags"],
        Integer[Array, " lags"],
    ]:
        """Return target and total actual/available counts for one position."""
        position_index = target_pos - 1
        valid_target = (
            (self.presentation[position_index] > 0)
            & self.avail_recalls[position_index]
            & self.condition_in_pool(position_index)
        )
        is_target_value = self.target_conditions[position_index] == self.target_value
        avail = self.lags_from_previous_to_target(target_pos, valid_target)
        actual = avail & jnp.any(recall_positions == target_pos)
        return (
            (actual & is_target_value).astype(int),
            actual.astype(int),
            (avail & is_target_value).astype(int),
            avail.astype(int),
        )

    def tabulate_lags(
        self, recall: Int_
    ) -> tuple[
        Integer[Array, " lags"],
        Integer[Array, " lags"],
        Integer[Array, " lags"],
        Integer[Array, " lags"],
    ]:
        """Return cumulative target and total actual/available lag counts."""
        recall_positions = self.item_study_positions[recall - 1]
        actual_target, actual_total, avail_target, avail_total = lax.map(
            lambda target_pos: self.tabulate_target_lags(target_pos, recall_positions),
            self.all_positions,
        )
        return (
            self.actual_target_lags + actual_target.sum(0),
            self.actual_total_lags + actual_total.sum(0),
            self.avail_target_lags + avail_target.sum(0),
            self.avail_total_lags + avail_total.sum(0),
        )

    def tabulate(
        self, recall: Int_, should_tabulate: Bool_
    ) -> "TargetEnrichmentTabulation":
        """Update state and optionally count the transition."""

        def _update_state() -> "TargetEnrichmentTabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "TargetEnrichmentTabulation":
                actual_target, actual_total, avail_target, avail_total = (
                    self.tabulate_lags(recall)
                )
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_target_lags=actual_target,
                    actual_total_lags=actual_total,
                    avail_target_lags=avail_target,
                    avail_total_lags=avail_total,
                )

            def _without_counts() -> "TargetEnrichmentTabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    target_conditions: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    target_value: int,
    target_values: Integer[Array, " target_values"],
    size: int = 3,
) -> tuple[
    Float[Array, " lags"],
    Float[Array, " lags"],
    Float[Array, " lags"],
    Float[Array, " lags"],
]:
    """Tabulate target and total actual/available lags for one trial."""
    target_values = jnp.asarray(target_values)
    has_recall = jnp.any(trial > 0)
    first_index = jnp.argmax(trial > 0)

    def tabulate_nonempty():
        init = TargetEnrichmentTabulation(
            presentation,
            target_conditions,
            target_value,
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
        return (
            tab.actual_target_lags,
            tab.actual_total_lags,
            tab.avail_target_lags,
            tab.avail_total_lags,
        )

    def tabulate_empty():
        lag_range = presentation.size - 1
        empty_lags = jnp.zeros(lag_range * 2 + 1, dtype=int)
        return empty_lags, empty_lags, empty_lags, empty_lags

    return lax.cond(has_recall, tabulate_nonempty, tabulate_empty)


def target_enrichment_crp(
    dataset: RecallDataset,
    target_field: str,
    target_value: int,
    target_values: Sequence[int] | Integer[Array, " target_values"],
    size: int = 3,
    source_field: Optional[str] = None,
    source_values: Optional[Sequence[int] | Integer[Array, " source_conditions"]] = None,
) -> Float[Array, " ... lags"]:
    """Compute lag-specific target enrichment relative to availability.

    Positive values mean actual transitions at a lag select ``target_value``
    more often than expected from the lag-specific available targets in
    ``target_values``.
    """
    if (source_field is None) != (source_values is None):
        raise ValueError("source_field and source_values must be provided together")

    target_values = jnp.asarray(target_values)
    should_tabulate = base_tabulation_mask(dataset)

    if source_field is not None:
        source_values = jnp.asarray(source_values)
        source_masks = source_tabulation_masks(dataset, source_field, source_values)
        should_tabulate = source_masks & should_tabulate

    def _enrichment_for_mask(mask):
        actual_target, actual_total, avail_target, avail_total = vmap(
            tabulate_trial, in_axes=(0, 0, 0, 0, None, None, None)
        )(
            dataset["recalls"],
            dataset["pres_itemnos"],
            dataset[target_field],  # type: ignore
            mask,
            target_value,
            target_values,
            size,
        )
        actual_target_sum = actual_target.sum(0)
        actual_total_sum = actual_total.sum(0)
        avail_target_sum = avail_target.sum(0)
        avail_total_sum = avail_total.sum(0)
        observed = jnp.where(
            actual_total_sum > 0, actual_target_sum / actual_total_sum, jnp.nan
        )
        baseline = jnp.where(
            avail_total_sum > 0, avail_target_sum / avail_total_sum, jnp.nan
        )
        return observed - baseline

    if source_field is not None:
        return vmap(_enrichment_for_mask)(should_tabulate)
    return _enrichment_for_mask(should_tabulate)


def plot_target_enrichment_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    target_field: str,
    target_value: int,
    target_values: Sequence[int],
    max_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    source_field: Optional[str] = None,
    source_values: Optional[Sequence[int]] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise lag-specific target enrichment."""
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if (source_field is None) != (source_values is None):
        raise ValueError("source_field and source_values must be provided together")

    condition_count = len(source_values) if source_field is not None else 0
    if labels is None:
        size_label_count = condition_count if condition_count > 1 else len(datasets)
        labels = [""] * size_label_count

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(
                target_enrichment_crp,
                static_argnames=(
                    "target_field",
                    "target_value",
                    "source_field",
                    "size",
                ),
            ),
            target_field=target_field,
            target_value=target_value,
            target_values=jnp.asarray(target_values),
            size=size,
            source_field=source_field,
            source_values=(
                jnp.asarray(source_values) if source_values is not None else None
            ),
        )
        if condition_count:
            for condition_index in range(condition_count):
                condition_subject_values = jnp.vstack(
                    [each[condition_index] for each in subject_values]
                )[:, lag_range - max_lag : lag_range + max_lag + 1]

                color_idx = data_index * condition_count + condition_index
                color = color_cycle[color_idx % len(color_cycle)]
                label = (
                    labels[color_idx]
                    if len(labels) > condition_count
                    else (
                        labels[condition_index]
                        if condition_count > 1
                        else labels[data_index]
                    )
                )
                plot_data(
                    axis,
                    lag_interval,
                    condition_subject_values,
                    label,
                    color,
                    confidence_level=confidence_level,
                )
        else:
            subject_values = jnp.vstack(subject_values)
            subject_values = subject_values[
                :, lag_range - max_lag : lag_range + max_lag + 1
            ]
            color = color_cycle[data_index % len(color_cycle)]
            plot_data(
                axis,
                lag_interval,
                subject_values,
                labels[data_index],
                color,
                confidence_level=confidence_level,
            )

    axis.axhline(0, color="black", linewidth=1, alpha=0.5)
    set_plot_labels(axis, "Lag", "Target Enrichment", contrast_name)
    return axis
