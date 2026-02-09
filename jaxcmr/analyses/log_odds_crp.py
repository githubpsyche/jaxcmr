"""Lag-CRP log-odds contrasts.

Reuses the tabulation machinery from ``crp`` (``SimpleTabulation`` for
no-repeat lists, ``Tabulation`` for repeated items) to accumulate
actual and available lag counts per subject, then applies a logit
transform and subtracts a reference lag to produce log-odds contrasts.

Notes
-----
- Probabilities are clamped to ``[epsilon, 1 - epsilon]`` before the
  logit to avoid infinities; lags with zero availability stay NaN.
- Subtracting the reference lag removes subject-level baseline
  differences, making the contrast more comparable across subjects.

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
    """Uniform-list CRP tabulator."""

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
    """Lag-CRP tabulation for a single no-repeat trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in ``1..L`` with ``0`` pads.
    list_length : int
        Study-list length ``L``.

    Returns
    -------
    SimpleTabulation
        Tabulation tracking actual and available transitions.

    """
    return lax.scan(
        lambda tabulation, recall: (tabulation.update(recall), None),
        SimpleTabulation(list_length, trial[0]),
        trial[1:],
    )[0]


def simple_crp(
    trials: Integer[Array, "trials recall_events"], list_length: int
) -> Float[Array, " lags"]:
    """Lag-CRP across multiple recall trials.

    Parameters
    ----------
    trials : Integer[Array, "trials recall_events"]
        Serial positions in ``1..L`` with ``0`` pads.
    list_length : int
        Study-list length ``L``.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length ``2*L - 1`` indexed by lag offset.

    """
    tabulated_trials = lax.map(lambda t: simple_tabulate_trial(t, list_length), trials)
    total_actual_transitions = jnp.sum(tabulated_trials.actual_transitions, axis=0)
    total_possible_transitions = jnp.sum(tabulated_trials.avail_transitions, axis=0)
    return total_actual_transitions / total_possible_transitions


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean vector to modify.
    i : Int_
        1-based index; ``0`` is a no-op sentinel.

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
    """Per-transition CRP state supporting repeated items."""

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
    """Actual and available lags for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Recall events as study positions.
    presentation : Integer[Array, " study_events"]
        Item identifiers at each study position.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def log_odds_crp(
    dataset: RecallDataset,
    reference_lag: int = 10,
    epsilon: float = 1e-6,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Lag-CRP log-odds relative to a reference lag.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    reference_lag : int
        Lag used as the zero-log-odds baseline.
    epsilon : float
        Probability clamp before the logit transform.
    size : int
        Maximum positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        Log-odds CRP relative to ``reference_lag``.

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
    """Plot subject-wise Lag-CRP log-odds with error bounds.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        Recall datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean masks selecting trials per dataset.
    max_lag : int
        Maximum lag to plot.
    color_cycle : list[str], optional
        Colors for each dataset.
    labels : Sequence[str], optional
        Legend labels for ``datasets``.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Matplotlib Axes to plot on.
    reference_lag : int
        Lag defining the zero-log-odds baseline.
    epsilon : float
        Probability clamp for the logit transform.
    size : int
        Maximum positions an item can occupy.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Axes with the Lag-CRP log-odds plot.

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
