"""
Lag-CRP log-odds contrasts.

Utilities to compute Lag-Conditional Response Probability (Lag-CRP) counts and
express them as log-odds differences relative to a reference lag. Serial lag ℓ
is defined by the study positions of successive recalls (ℓ = Y - X for recalls
at positions X then Y). All outputs retain the standard Lag-CRP axis
conventions—length ``2*L - 1`` with index ``i`` corresponding to lag
``ℓ = i - (L - 1)``.
"""

__all__ = [
    "SimpleTabulation",
    "simple_tabulate_trial",
    "simple_crp",
    "set_false_at_index",
    "Tabulation",
    "tabulate_trial",
    "log_odds_crp",
    "plot_log_odds_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
from ..repetition import all_study_positions
from ..helpers import apply_by_subject

from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


class SimpleTabulation(Pytree):
    """
    Uniform-list CRP tabulator.

    Assumes:
        - `trial[0] > 0` (first recall exists).
        - Subsequent zeros are pads; ignored.

    Behavior:
        - Marks the first recall's item as unavailable for subsequent transitions.
        - Bounds-checks choices to [1..L]; others are ignored.
    """

    def __init__(self, list_length: int, first_recall: Int_):
        self.lag_range = list_length - 1
        self.all_items = jnp.arange(1, list_length + 1, dtype=int)
        self.actual_transitions = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_transitions = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_items = jnp.ones(list_length, dtype=bool)
        self.avail_items = self.avail_items.at[first_recall - 1].set(False)
        self.previous_item = first_recall

    def _update(self, current_item: Int_) -> "SimpleTabulation":
        "Tabulate actual and possible serial lags of current from previous item."
        actual_lag = current_item - self.previous_item + self.lag_range
        all_lags = self.all_items - self.previous_item + self.lag_range

        return self.replace(
            previous_item=current_item,
            avail_items=self.avail_items.at[current_item - 1].set(False),
            avail_transitions=self.avail_transitions.at[all_lags].add(self.avail_items),
            actual_transitions=self.actual_transitions.at[actual_lag].add(1),
        )

    def update(self, choice: Int_) -> "SimpleTabulation":
        "Tabulate a transition if the choice is non-zero (i.e., a valid item)."
        return lax.cond(choice > 0, lambda: self._update(choice), lambda: self)


def simple_tabulate_trial(
    trial: Integer[Array, " recall_events"], list_length: int
) -> SimpleTabulation:
    """Compute Lag-CRP tabulation for a single trial with no repeated items.

    Args:
        trial: int array [n_recall_events]; serial positions in 1..L with 0 pads.
        list_length: Study-list length ``L``.

    Returns:
        SimpleTabulation tracking actual and available transitions.

    Notes:
        * Ignores zeros (pads) and out-of-range positions.
        * Divides by available transitions; zero denominators yield NaN.
    """
    return lax.scan(
        lambda tabulation, recall: (tabulation.update(recall), None),
        SimpleTabulation(list_length, trial[0]),
        trial[1:],
    )[0]


def simple_crp(
    trials: Integer[Array, "trials recall_events"], list_length: int
) -> Float[Array, " lags"]:
    """Compute Lag-CRP across multiple recall trials.

    Args:
        trials: int array [n_trials, n_recall_events]; serial positions in 1..L, 0 pads.
        list_length: L.

    Returns:
        1-D float array of length (2*L - 1), indexed by lag offset (lag + (L - 1)).
    """
    tabulated_trials = lax.map(lambda t: simple_tabulate_trial(t, list_length), trials)
    total_actual_transitions = jnp.sum(tabulated_trials.actual_transitions, axis=0)
    total_possible_transitions = jnp.sum(tabulated_trials.avail_transitions, axis=0)
    return total_actual_transitions / total_possible_transitions


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


class Tabulation(Pytree):
    """
    Maintains per-transition state for CRP with repeats.

    State:
        - previous_positions: study positions of previously recalled item
        - avail_recalls: boolean [L], study positions still available
        - actual_lags, avail_lags: int [2*L - 1], accumulated

    Update steps executed on each valid recall (in order):
        1) Previous item positions -> `previous_positions`.
        2) Available positions -> `avail_recalls = available_recalls_after(recall)`.
        3) Actual lags: union of unique lags from `previous_positions` to the
           current item's study positions (`tabulate_actual_lags`).
        4) Available lags: union of unique lags from each `previous_positions`
           to all currently available positions (`tabulate_available_lags`).

    Conventions:
        - Zeros in `recall_study_positions` are padding; safely ignored.
        - Indices are 1-based for positions; internal arrays use zero-based.
    """

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
        """
        Clear availability for all study positions of `recall`.
        Safe with padding: zeros are ignored.
        """
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """
        One-hot(-ish) vector of unique lags from each previous study position to `recall_pos`.
        Returns boolean union (unique lag set); use `count_mode="pairwise"` alternative
        (not implemented here) to count all pair combinations.
        """

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - prev + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulates the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identifies recallable lag transitions from the specified study position."
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Union of lags from each previous position to all currently available positions."
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self, recall: Int_) -> Bool:
        "Only consider transitions from items with study positions that have not been recalled yet."

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

    def tabulate(self, recall: Int_) -> "Tabulation":
        "Tabulates actual and possible serial lags of current from previous item."
        return lax.cond(
            self.should_tabulate(recall),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
            ),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial."""

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def log_odds_crp(
    dataset: RecallDataset,
    reference_lag: int = 10,
    epsilon: float = 1e-6,
    size: int = 3,
) -> Float[Array, " lags"]:
    """
    Returns Lag-CRP log-odds relative to a reference lag.

    Args:
        dataset: Recall dataset containing ``recalls`` and ``pres_itemnos``.
        reference_lag: Lag (ℓ) used as the zero-log-odds baseline.
        epsilon: Lower/upper bound for probabilities before taking the logit.
        size: Max number of study positions an item can occupy (compile-time constant).
    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"], dataset["pres_itemnos"], size
    )
    total_actual = actual.sum(0)
    total_possible = possible.sum(0)
    probabilities = total_actual / total_possible
    # Lags with zero availability stay NaN here so they never fabricate odds.
    clipped = jnp.clip(probabilities, epsilon, 1 - epsilon)
    # Clipping avoids ±inf when actual counts hit the bounds but keeps NaNs intact.
    logits = jnp.log(clipped) - jnp.log1p(-clipped)
    lag_range = dataset["pres_itemnos"].shape[1] - 1
    reference_index = reference_lag + lag_range
    reference_value = logits[reference_index]
    # Subtracting the reference lag removes subject-level baseline differences.
    return logits - reference_value


def plot_log_odds_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    reference_lag: int = 10,
    epsilon: float = 1e-6,
    size: int = 3,
    confidence_level: float = 0.95,

) -> Axes:
    """
    Plot subject-wise Lag-CRP log-odds and their mean ± error.

    Args:
        datasets: Datasets containing trial data to be plotted.
        trial_masks: Masks to filter trials in datasets.
        max_lag: Maximum lag to plot.
        color_cycle: List of colors for plotting each dataset.
        labels: Names for each dataset for legend, optional.
        contrast_name: Name of contrast for legend labeling, optional.
        axis: Existing matplotlib Axes to plot on, optional.
        reference_lag: Lag that defines the zero log-odds baseline.
        epsilon: Probability clamp used inside the logit transform.
        size: Maximum number of study positions an item can be presented at.
        confidence_level: Confidence level for the bounds.


    Returns:
        Matplotlib Axes with the Lag-CRP plot.

    Notes:
        - `datasets` must contain 'recalls', 'pres_itemnos', 'listLength'.
        - `trial_masks` filters trials; lengths must match datasets.
        - Color cycle wraps if more datasets than colors.
    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(datasets, trial_masks, color_cycle, axis)


    if labels is None:
        labels = [""] * len(datasets)



    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = (jnp.max(data["listLength"]) - 1).item()
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(
                log_odds_crp,
                static_argnames=(
                    "reference_lag",
                    "size",
                ),
            ),
            reference_lag=reference_lag,
            epsilon=epsilon,
            size=size,
        )
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
            confidence_level=confidence_level
        )

    set_plot_labels(
        axis,
        "Lag",
        f"Log-Odds CRP (vs lag {reference_lag})",
        contrast_name,
    )
    return axis
