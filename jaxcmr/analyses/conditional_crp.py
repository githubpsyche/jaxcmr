"""Lag-CRP with conditional transition filtering.

Extends ``crp.Tabulation`` with a per-event ``_should_tabulate`` mask
so that only a subset of transitions contribute to actual and available
lag counts, while all recalls still update availability tracking. This
supports analyses that condition on properties of the transition (e.g.,
lag direction, item category, or repetition status).

The ``Tabulation`` class shares its state layout and update logic with
``crp.Tabulation``; see that module for conventions and update steps.

"""

__all__ = [
    "set_false_at_index",
    "Tabulation",
    "tabulate_trial",
    "crp",
    "plot_crp",
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


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean availability vector.
    i : Int_
        1-based index to clear; ``0`` is a no-op sentinel.

    Returns
    -------
    tuple[Bool[Array, " positions"], None]
        Updated vector and ``None``.

    """

    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class Tabulation(Pytree):
    """Per-transition state for conditional Lag-CRP."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for study positions of ``recall``.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Bool[Array, " positions"]
            Updated availability vector.

        """
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """Compute unique lags from previous positions to ``recall_pos``.

        Parameters
        ----------
        recall_pos : Int_
            Study position of the current recall.

        Returns
        -------
        Bool[Array, " positions"]
            Boolean lag vector with True at each unique lag.

        """

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - prev + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        """Tabulate actual transition lags for a recall event.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated actual lag counts.

        """
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        """Identify recallable lag transitions from ``pos``.

        Parameters
        ----------
        pos : Int_
            Study position of the previous recall.

        Returns
        -------
        Bool[Array, " lags"]
            Boolean lag vector of available transitions.

        """
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        """Union of lags from each previous position to available items.

        Returns
        -------
        Integer[Array, " lags"]
            Accumulated available lag counts.

        """
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def is_valid_recall(self, recall: Int_) -> Bool:
        """Return True when recall maps to an available position.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).

        Returns
        -------
        Bool
            Whether the recall is valid.

        """

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def tabulate(self, recall: Int_, should_tabulate: Bool_) -> "Tabulation":
        """Update state and optionally count the transition.

        Parameters
        ----------
        recall : Int_
            Recalled item (1-indexed study position).
        should_tabulate : Bool_
            Whether to count this transition.

        Returns
        -------
        Tabulation
            Updated tabulation state.

        """

        def _update_state() -> "Tabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_lags=self.tabulate_actual_lags(recall),
                    avail_lags=self.tabulate_available_lags(),
                )

            def _without_counts() -> "Tabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


class TargetCondTabulation(Pytree):
    """Per-transition state for target-condition Lag-CRP."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        target_conditions: Integer[Array, " study_events"],
        target_values: Integer[Array, " target_conditions"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.size = size
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.presentation = presentation
        self.valid_targets = presentation > 0
        self.target_conditions = target_conditions
        self.target_values = target_values
        self.condition_indices = jnp.arange(target_values.size, dtype=int)
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(
            (target_values.size, self.lag_range * 2 + 1), dtype=int
        )
        self.avail_lags = jnp.zeros(
            (target_values.size, self.lag_range * 2 + 1), dtype=int
        )

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
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

    def tabulate_condition_lags(
        self,
        condition_index: Int_,
        target_pos: Int_,
        recall_positions: Integer[Array, " size"],
    ) -> tuple[Integer[Array, " lags"], Integer[Array, " lags"]]:
        """Return actual and available lags for one condition and target."""
        position_index = target_pos - 1
        condition_value = self.target_values[condition_index]
        valid_target = (
            self.valid_targets[position_index]
            & self.avail_recalls[position_index]
            & (self.target_conditions[position_index] == condition_value)
        )
        avail = self.lags_from_previous_to_target(target_pos, valid_target)
        actual = avail & jnp.any(recall_positions == target_pos)
        return actual.astype(int), avail.astype(int)

    def tabulate_target_lags(
        self, target_pos: Int_, recall_positions: Integer[Array, " size"]
    ) -> tuple[
        Integer[Array, " target_conditions lags"],
        Integer[Array, " target_conditions lags"],
    ]:
        """Return condition-specific lag counts for one target position."""
        return lax.map(
            lambda condition_index: self.tabulate_condition_lags(
                condition_index, target_pos, recall_positions
            ),
            self.condition_indices,
        )

    def tabulate_lags(
        self, recall: Int_
    ) -> tuple[
        Integer[Array, " target_conditions lags"],
        Integer[Array, " target_conditions lags"],
    ]:
        """Return cumulative actual and available target-condition lags."""
        recall_positions = self.item_study_positions[recall - 1]
        actual, avail = lax.map(
            lambda target_pos: self.tabulate_target_lags(target_pos, recall_positions),
            self.all_positions,
        )
        return self.actual_lags + actual.sum(0), self.avail_lags + avail.sum(0)

    def tabulate(self, recall: Int_, should_tabulate: Bool_) -> "TargetCondTabulation":
        """Update state and optionally count the transition."""

        def _update_state() -> "TargetCondTabulation":
            new_previous_positions = self.item_study_positions[recall - 1]
            new_avail_recalls = self.available_recalls_after(recall)

            def _with_counts() -> "TargetCondTabulation":
                actual_lags, avail_lags = self.tabulate_lags(recall)
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                    actual_lags=actual_lags,
                    avail_lags=avail_lags,
                )

            def _without_counts() -> "TargetCondTabulation":
                return self.replace(
                    previous_positions=new_previous_positions,
                    avail_recalls=new_avail_recalls,
                )

            return lax.cond(should_tabulate, _with_counts, _without_counts)

        return lax.cond(self.is_valid_recall(recall), _update_state, lambda: self)


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall sequence as within-list positions.
    presentation : Integer[Array, " study_events"]
        Study presentation order for the trial.
    should_tabulate : Bool[Array, " recall_events"]
        Boolean mask; True counts the transition.
    size : int, optional
        Max study positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(
        1, trial.size, lambda i, t: t.tabulate(trial[i], should_tabulate[i]), init
    )
    return tab.actual_lags, tab.avail_lags


def tabulate_target_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    target_conditions: Integer[Array, " study_events"],
    should_tabulate: Bool[Array, " recall_events"],
    target_values: Integer[Array, " target_conditions"],
    size: int = 3,
) -> tuple[
    Float[Array, " target_conditions lags"],
    Float[Array, " target_conditions lags"],
]:
    """Tabulate actual and available target-condition lags for one trial."""
    target_values = jnp.asarray(target_values)
    has_recall = jnp.any(trial > 0)
    first_index = jnp.argmax(trial > 0)

    def tabulate_nonempty():
        init = TargetCondTabulation(
            presentation,
            target_conditions,
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
        return tab.actual_lags, tab.avail_lags

    def tabulate_empty():
        lag_range = presentation.size - 1
        empty_lags = jnp.zeros((target_values.size, lag_range * 2 + 1), dtype=int)
        return empty_lags, empty_lags

    return lax.cond(has_recall, tabulate_nonempty, tabulate_empty)


def source_tabulation_masks(
    dataset: RecallDataset,
    source_field: str,
    source_values: Integer[Array, " source_conditions"],
) -> Bool[Array, " source_conditions trials recall_events"]:
    """Build per-event masks for transitions from source values."""
    recalls = dataset["recalls"]
    field = dataset[source_field]  # type: ignore
    previous_recalls = recalls[:, :-1]
    current_recalls = recalls[:, 1:]
    valid_transitions = (previous_recalls > 0) & (current_recalls > 0)

    transition_indices = jnp.clip(previous_recalls - 1, 0, field.shape[1] - 1)
    transition_values_by_event = jnp.take_along_axis(field, transition_indices, axis=1)

    def mask_for_value(source_value):
        mask = jnp.zeros_like(recalls, dtype=bool)
        mask = mask.at[:, 1:].set(
            valid_transitions & (transition_values_by_event == source_value)
        )
        return mask

    return vmap(mask_for_value)(source_values)


def base_tabulation_mask(dataset: RecallDataset) -> Bool[Array, " trials recall_events"]:
    """Return the dataset mask, or all True when no mask is supplied."""
    if "_should_tabulate" in dataset:
        return jnp.asarray(dataset["_should_tabulate"], dtype=bool)  # type: ignore
    return jnp.ones_like(dataset["recalls"], dtype=bool)


def crp(
    dataset: RecallDataset,
    size: int = 3,
    source_field: Optional[str] = None,
    source_values: Optional[Sequence[int] | Integer[Array, " source_conditions"]] = None,
    target_field: Optional[str] = None,
    target_values: Optional[Sequence[int] | Integer[Array, " target_conditions"]] = None,
) -> Float[Array, " ... lags"]:
    """Compute conditional Lag-CRP with repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Dataset with ``recalls``, ``pres_itemnos``, and
        ``_should_tabulate``.
    size : int, optional
        Max study positions an item can occupy.
    source_field : str, optional
        Dataset key used to filter transitions by the previous recall.
    source_values : Sequence[int], optional
        Source values to tabulate separately.
    target_field : str, optional
        Dataset key used to split target denominators.
    target_values : Sequence[int], optional
        Target values to tabulate separately.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1; NaN where denominator is zero.

    """
    if (source_field is None) != (source_values is None):
        raise ValueError("source_field and source_values must be provided together")
    if (target_field is None) != (target_values is None):
        raise ValueError("target_field and target_values must be provided together")

    should_tabulate = base_tabulation_mask(dataset)

    if source_field is not None:
        source_values = jnp.asarray(source_values)
        source_masks = source_tabulation_masks(dataset, source_field, source_values)
        should_tabulate = source_masks & should_tabulate

    if target_field is not None:
        target_values = jnp.asarray(target_values)
        if source_field is not None and source_values.size != 1:
            raise ValueError("source_values must have length 1 when target_field is set")
        if source_field is not None:
            should_tabulate = should_tabulate[0]
        actual, possible = vmap(
            tabulate_target_trial, in_axes=(0, 0, 0, 0, None, None)
        )(
            dataset["recalls"],
            dataset["pres_itemnos"],
            dataset[target_field],  # type: ignore
            should_tabulate,
            target_values,
            size,
        )
        actual_sum = actual.sum(0)
        possible_sum = possible.sum(0)
        return jnp.where(possible_sum > 0, actual_sum / possible_sum, jnp.nan)

    def _crp_for_mask(mask):
        actual, possible = vmap(tabulate_trial, in_axes=(0, 0, 0, None))(
            dataset["recalls"],
            dataset["pres_itemnos"],
            mask,
            size,
        )
        actual_sum = actual.sum(0)
        possible_sum = possible.sum(0)
        return jnp.where(possible_sum > 0, actual_sum / possible_sum, jnp.nan)

    if source_field is not None:
        return vmap(_crp_for_mask)(should_tabulate)
    return _crp_for_mask(should_tabulate)


def plot_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 4,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    source_field: Optional[str] = None,
    source_values: Optional[Sequence[int]] = None,
    target_field: Optional[str] = None,
    target_values: Optional[Sequence[int]] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise conditional Lag-CRP with error bounds.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Datasets containing trial data to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Masks to filter trials in datasets.
    max_lag : int, optional
        Maximum lag to plot.
    color_cycle : list[str], optional
        Colors for plotting each dataset.
    labels : Sequence[str], optional
        Legend labels for each dataset.
    contrast_name : str, optional
        Legend title for contrasts.
    axis : Axes, optional
        Existing matplotlib Axes to plot on.
    size : int, optional
        Max study positions an item can occupy.
    source_field : str, optional
        Dataset key used to filter transitions by the previous recall.
    source_values : Sequence[int], optional
        Source values to tabulate separately.
    target_field : str, optional
        Dataset key used to split target denominators.
    target_values : Sequence[int], optional
        Target values to tabulate separately.
    confidence_level : float, optional
        Confidence level for the bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the Lag-CRP plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if (source_field is None) != (source_values is None):
        raise ValueError("source_field and source_values must be provided together")
    if (target_field is None) != (target_values is None):
        raise ValueError("target_field and target_values must be provided together")
    if target_field is not None and source_field is not None and len(source_values) != 1:
        raise ValueError("source_values must have length 1 when target_field is set")

    if target_field is not None:
        condition_count = len(target_values)
    elif source_field is not None:
        condition_count = len(source_values)
    else:
        condition_count = 0

    if labels is None:
        size_label_count = condition_count if condition_count > 1 else len(datasets)
        labels = [""] * size_label_count

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = jnp.max(data["listLength"][trial_masks[data_index]]) - 1
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(crp, static_argnames=("source_field", "target_field", "size")),
            size=size,
            source_field=source_field,
            source_values=(
                jnp.asarray(source_values) if source_values is not None else None
            ),
            target_field=target_field,
            target_values=(
                jnp.asarray(target_values) if target_values is not None else None
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

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
